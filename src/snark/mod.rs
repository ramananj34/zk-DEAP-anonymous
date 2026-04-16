use serde::{Deserialize, Serialize}; //Serialization/deserialization
use serde_with::{serde_as, Bytes}; //Serialization/deserialization
use curve25519_dalek_ng::{ristretto::*, scalar::Scalar, traits::Identity, constants::RISTRETTO_BASEPOINT_POINT}; //Curve
use halo2_proofs::{arithmetic::Field, halo2curves::{bn256::{Bn256, Fr as Halo2Fr, G1Affine}, ff::PrimeField}, circuit::{Layouter, SimpleFloorPlanner, Value}, plonk::{Advice, Circuit, Column, ConstraintSystem, Error as Halo2Error, Expression, Fixed, Selector, create_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey, VerifyingKey, Instance}, poly::{Rotation, commitment::Params, kzg::{commitment::{KZGCommitmentScheme, ParamsKZG}, multiopen::{ProverSHPLONK, VerifierSHPLONK}, strategy::SingleStrategy}}, transcript::{Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer}}; //Halo2
use merlin::Transcript; //Transcript
use frost_ristretto255 as frost; //Frost
use ed25519_dalek::{SigningKey, VerifyingKey as ed_vf, Signature, Signer, Verifier}; //ECDS
use zeroize::Zeroize; //Safe deletion
use rand::rngs::OsRng; //Random
use rand::RngCore; //Random
use std::collections::{HashMap, HashSet}; //Data structures
use std::fs::File; //File operations
use std::io::BufReader; //File operations
use std::path::Path; //File operations

use crate::common::{SerCompressed, SerScalar, AggError, VerifiedCiphertext, VerifiedPartial, PartialDecryption, PROTOCOL_VERSION, MAX_DEVICES, MAX_STORED_PROOFS, MAX_CLOCK_SKEW,MAX_NONCES_PER_DEVICE, PROOF_EXPIRY, timestamp, frost_to_point, check_rate};

const MAX_PROOF_SIZE: usize = 8192;
const _HALO2_K: u32 = 8;
const PARAMS_PATH: &str = "./trusted_setup/kzg_bn254_8.params";
const TRACE_LENGTH: usize = 64;
const MIMC_ROUNDS: usize = TRACE_LENGTH - 1; //63 rounds, one per transition

//Public MiMC round constants (nothing-up-my-sleeve, from BLAKE3)
fn mimc_round_constants() -> [Halo2Fr; TRACE_LENGTH] {
    let mut rc = [Halo2Fr::ZERO; TRACE_LENGTH];
    for i in 0..TRACE_LENGTH {
        let label = format!("zk-DEAP-MiMC-rc-{}", i);
        let h = blake3::hash(label.as_bytes());
        let bytes = h.as_bytes();
        //Pad to 32 bytes and reduce mod r via from_repr
        let mut repr = [0u8; 32];
        repr[..16].copy_from_slice(&bytes[..16]); //keep low 128 bits, rest zero
        rc[i] = Halo2Fr::from_repr(repr).unwrap_or(Halo2Fr::ZERO);
    }
    rc
}

//MiMC_63 hash: x_{i+1} = (x_i + s + eps + b + rc_i)^5
fn mimc_hash(v_f: Halo2Fr, s: Halo2Fr, eps: Halo2Fr, b: Halo2Fr) -> Halo2Fr {
    let rc = mimc_round_constants();
    let mut x = v_f;
    for i in 0..MIMC_ROUNDS {
        let sum = x + s + eps + b + rc[i];
        let sq = sum * sum;
        x = sq * sq * sum; //x^5
    }
    x
}

//Ristretto Scalar to BN254 Fr. q < r, so direct from_repr always succeeds.
fn scalar_to_fr(s: &Scalar) -> Halo2Fr {
    Halo2Fr::from_repr(s.to_bytes()).unwrap_or(Halo2Fr::ZERO)
}

//Encode a 32-byte compressed Ristretto point as Fr via XOR-fold to 128 bits.
fn point_to_fr(bytes: &[u8; 32]) -> Halo2Fr {
    let mut folded = [0u8; 32];
    for i in 0..16 { folded[i] = bytes[i] ^ bytes[i + 16]; }
    Halo2Fr::from_repr(folded).unwrap_or(Halo2Fr::ZERO)
}

//q (Ristretto scalar order) embedded as Fr. Since q < r, just reduce (-1_q + 1).
fn q_mod_r() -> Halo2Fr {
    scalar_to_fr(&(-Scalar::one())) + Halo2Fr::ONE
}

//Load KZG parameters
pub fn load_kzg_params() -> Result<ParamsKZG<Bn256>, AggError> {
    let path = Path::new(PARAMS_PATH);
    if !path.exists() { return Err(AggError::CryptoError(format!("KZG params not found at: {}\nRun setup first!", PARAMS_PATH))); }
    let file = File::open(path).map_err(|e| AggError::CryptoError(format!("Failed to open params: {}", e)))?;
    let mut reader = BufReader::new(file);
    ParamsKZG::<Bn256>::read(&mut reader).map_err(|e| AggError::CryptoError(format!("Failed to read params: {}", e)))
}

//Halo2 circuit configuration
#[derive(Clone, Debug)]
struct BinaryConfig {
    advice: [Column<Advice>; 4],
    rc_col: Column<Fixed>,
    s_mimc: Selector,
    s_row0: Selector,
    instance: [Column<Instance>; 4],
}
//Circuit proving state is binary
#[derive(Clone, Debug)]
struct BinaryCircuit {
    state: Value<Halo2Fr>,
    v_f: Value<Halo2Fr>,
    epsilon: Value<Halo2Fr>,
    blinding: Value<Halo2Fr>,
}
impl Circuit<Halo2Fr> for BinaryCircuit {
    type Config = BinaryConfig;
    type FloorPlanner = SimpleFloorPlanner;
    fn without_witnesses(&self) -> Self {
        Self {
            state: Value::unknown(),
            v_f: Value::unknown(),
            epsilon: Value::unknown(),
            blinding: Value::unknown(),
        }
    }
    fn configure(meta: &mut ConstraintSystem<Halo2Fr>) -> Self::Config {
        let advice = [meta.advice_column(), meta.advice_column(), meta.advice_column(), meta.advice_column()];
        let rc_col = meta.fixed_column();
        let s_mimc = meta.selector();
        let s_row0 = meta.selector();
        let instance = [meta.instance_column(), meta.instance_column(), meta.instance_column(), meta.instance_column()];
        for col in &advice { meta.enable_equality(*col); }
        for col in &instance { meta.enable_equality(*col); }
        let q_const_val = q_mod_r();
        //MiMC transition + constancy (rows 0..62)
        meta.create_gate("mimc_and_constancy", |meta| {
            let sel = meta.query_selector(s_mimc);
            let s = meta.query_advice(advice[0], Rotation::cur());
            let s_nx = meta.query_advice(advice[0], Rotation::next());
            let m = meta.query_advice(advice[1], Rotation::cur());
            let m_nx = meta.query_advice(advice[1], Rotation::next());
            let eps = meta.query_advice(advice[2], Rotation::cur());
            let eps_nx= meta.query_advice(advice[2], Rotation::next());
            let b= meta.query_advice(advice[3], Rotation::cur());
            let b_nx= meta.query_advice(advice[3], Rotation::next());
            let rc= meta.query_fixed(rc_col, Rotation::cur());
            let sum = m.clone() + s.clone() + eps.clone() + b.clone() + rc;
            let sum_sq = sum.clone() * sum.clone();
            let sum_5 = sum_sq.clone() * sum_sq * sum;
            vec![
                sel.clone() * (m_nx - sum_5), //MiMC round (degree 5)
                sel.clone() * (s_nx - s), //s constancy
                sel.clone() * (eps_nx - eps), //eps constancy
                sel * (b_nx - b), //b constancy
            ]
        });
        //Binary + Schnorr linear factored (row 0 only)
        meta.create_gate("binary_and_schnorr", |meta| {
            let sel = meta.query_selector(s_row0);
            let s  = meta.query_advice(advice[0], Rotation::cur());
            let vf = meta.query_advice(advice[1], Rotation::cur()); //m at row 0 = v_f
            let resp_f = meta.query_instance(instance[2], Rotation::cur());
            let c_f    = meta.query_instance(instance[3], Rotation::cur());
            let q_const = Expression::Constant(q_const_val);
            let one = Expression::Constant(Halo2Fr::ONE);
            //Factored Schnorr: (resp_f - v_f - c_f*s) * (resp_f - v_f - c_f*s + q*s) = 0. Covers both wrap and no-wrap cases since r > 2q.
            let diff = resp_f - vf - c_f * s.clone();
            let diff_plus_qs = diff.clone() + q_const * s.clone();
            vec![
                sel.clone() * s.clone() * (s - one), //binary (degree 2)
                sel * diff * diff_plus_qs, //schnorr factored (degree 2)
            ]
        });
        BinaryConfig { advice, rc_col, s_mimc, s_row0, instance }
    }
    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Halo2Fr>) -> Result<(), Halo2Error> {
        let rc = mimc_round_constants();
        //Compute MiMC chain in Value space (so it works during both keygen and proving)
        let mut mimc_chain: Vec<Value<Halo2Fr>> = Vec::with_capacity(TRACE_LENGTH);
        mimc_chain.push(self.v_f);
        for i in 0..MIMC_ROUNDS {
            let rc_i = rc[i];
            let next = mimc_chain[i].zip(self.state).zip(self.epsilon).zip(self.blinding).map(|(((m, s), e), b)| {let sum = m + s + e + b + rc_i;let sq = sum * sum;sq * sq * sum});
            mimc_chain.push(next);
        }
        let (m_final_cell, eps_cell) = layouter.assign_region(|| "mimc_chain", |mut region| {
            //Enable row-0 gates
            config.s_row0.enable(&mut region, 0)?;
            //Enable s_mimc at rows 0..62; assign rc at each transition row
            for i in 0..MIMC_ROUNDS {
                config.s_mimc.enable(&mut region, i)?;
                region.assign_fixed(|| format!("rc_{}", i), config.rc_col, i, || Value::known(rc[i]))?;
            }
            //Last row (63) has no outgoing transition; assign 0 for completeness.
            region.assign_fixed(|| "rc_last", config.rc_col, TRACE_LENGTH - 1, || Value::known(Halo2Fr::ZERO))?;
            //Assign all advice cells across 64 rows
            let mut m_final = None;
            let mut eps_first = None;
            for i in 0..TRACE_LENGTH {
                region.assign_advice(|| format!("s_{}", i), config.advice[0], i, || self.state)?;
                let m_cell = region.assign_advice(|| format!("m_{}", i), config.advice[1], i, || mimc_chain[i])?;
                if i == TRACE_LENGTH - 1 { m_final = Some(m_cell); }
                let e_cell = region.assign_advice(|| format!("eps_{}", i), config.advice[2], i, || self.epsilon)?;
                if i == 0 { eps_first = Some(e_cell); }
                region.assign_advice(|| format!("b_{}", i), config.advice[3], i, || self.blinding)?;
            }
            Ok((m_final.unwrap(), eps_first.unwrap()))
        })?;

        //Bind public inputs: V_commit = m[63], epsilon = eps[0] (resp_f and c_f are bound via query_instance inside the row-0 gate, so no constrain_instance is needed for them.)
        layouter.constrain_instance(m_final_cell.cell(), config.instance[0], 0)?; //V_commit
        layouter.constrain_instance(eps_cell.cell(), config.instance[1], 0)?;      //epsilon
        Ok(())
    }
}
//Halo2 setup
#[derive(Clone)]
pub struct Halo2Setup {
    params: ParamsKZG<Bn256>,
    pk: ProvingKey<G1Affine>,
    vk: VerifyingKey<G1Affine>
}

//Setup Halo2
pub fn setup_halo2() -> Result<Halo2Setup, AggError> {
    let params = load_kzg_params()?;
    let empty_circuit = BinaryCircuit {state: Value::unknown(),v_f: Value::unknown(),epsilon: Value::unknown(),blinding: Value::unknown()};
    let vk = keygen_vk(&params, &empty_circuit).map_err(|e| AggError::CryptoError(format!("VK gen failed: {:?}", e)))?;
    let pk = keygen_pk(&params, vk.clone(), &empty_circuit).map_err(|e| AggError::CryptoError(format!("PK gen failed: {:?}", e)))?;
    Ok(Halo2Setup { params, pk, vk })
}

//ElGamal correctness proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElGamalProof {
    pub commit_r: SerCompressed,
    pub commit_s: SerCompressed,
    pub commit_p: SerCompressed,
    pub resp_r: SerScalar,
    pub resp_state: SerScalar,
    pub pedersen_commit: SerCompressed,
}
impl ElGamalProof {
    //Schnorr prove with caller-supplied nonce v. V_commit is bound into transcript before challenge derivation. Returns (proof, challenge).
    pub fn prove_with_nonce(state: u8, r: &Scalar, v: &Scalar, c1: &RistrettoPoint, c2: &RistrettoPoint, h: &RistrettoPoint, dev_id: u32, ts: u64, v_commit_bytes: &[u8; 16]) -> (Self, Scalar) {
        let g = RISTRETTO_BASEPOINT_POINT;
        let s_scalar = Scalar::from(state as u64);
        let pedersen = g * s_scalar + h * r;
        let mut w = Scalar::random(&mut OsRng);
        let cr = g * w;
        let cs = g * v + h * w;
        let cp = g * v + h * w;
        let mut t = Transcript::new(b"elgamal-pedersen-v2");
        t.append_message(b"protocol_version", &[PROTOCOL_VERSION]);
        t.append_u64(b"device", dev_id as u64);
        t.append_u64(b"timestamp", ts);
        t.append_message(b"c1", c1.compress().as_bytes());
        t.append_message(b"c2", c2.compress().as_bytes());
        t.append_message(b"pedersen", pedersen.compress().as_bytes());
        t.append_message(b"R", cr.compress().as_bytes());
        t.append_message(b"S", cs.compress().as_bytes());
        t.append_message(b"P", cp.compress().as_bytes());
        t.append_message(b"v_commit", v_commit_bytes);
        let mut cb = [0u8; 64];
        t.challenge_bytes(b"challenge", &mut cb);
        let c = Scalar::from_bytes_mod_order_wide(&cb);
        let proof = Self {
            commit_r: cr.compress().into(),
            commit_s: cs.compress().into(),
            commit_p: cp.compress().into(),
            resp_r: (w + c * r).into(),
            resp_state: (v + c * s_scalar).into(),
            pedersen_commit: pedersen.compress().into(),
        };
        w.zeroize();
        (proof, c)
    }
    pub fn verify(&self, c1: &RistrettoPoint, c2: &RistrettoPoint, h: &RistrettoPoint, dev_id: u32, ts: u64, v_commit_bytes: &[u8; 16]) -> bool {
        let g = RISTRETTO_BASEPOINT_POINT;
        let (Some(cr), Some(cs), Some(cp), Some(pc)) = (self.commit_r.0.decompress(),self.commit_s.0.decompress(),self.commit_p.0.decompress(),self.pedersen_commit.0.decompress()) else { return false };
        let mut t = Transcript::new(b"elgamal-pedersen-v2");
        t.append_message(b"protocol_version", &[PROTOCOL_VERSION]);
        t.append_u64(b"device", dev_id as u64);
        t.append_u64(b"timestamp", ts);
        t.append_message(b"c1", c1.compress().as_bytes());
        t.append_message(b"c2", c2.compress().as_bytes());
        t.append_message(b"pedersen", self.pedersen_commit.0.as_bytes());
        t.append_message(b"R", self.commit_r.0.as_bytes());
        t.append_message(b"S", self.commit_s.0.as_bytes());
        t.append_message(b"P", self.commit_p.0.as_bytes());
        t.append_message(b"v_commit", v_commit_bytes);
        let mut cb = [0u8; 64];
        t.challenge_bytes(b"challenge", &mut cb);
        let c = Scalar::from_bytes_mod_order_wide(&cb);
        let chk1 = g * self.resp_r.0 == cr + c1 * c;
        let chk2 = g * self.resp_state.0 + h * self.resp_r.0 == cs + c2 * c;
        let chk3 = g * self.resp_state.0 + h * self.resp_r.0 == cp + pc * c;
        chk1 && chk2 && chk3
    }
}

//Device proof with Halo2
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProof {
    pub device_id: u32,
    pub timestamp: u64,
    pub elgamal_c1: SerCompressed,
    pub elgamal_c2: SerCompressed,
    pub elgamal_proof: ElGamalProof,
    pub v_commit: [u8; 32], //MiMC output, embedded as Fr
    pub resp_f: [u8; 32], //Schnorr response embedded as Fr
    pub c_f: [u8; 32], //challenge embedded as Fr
    pub halo2_proof: Vec<u8>,
    #[serde_as(as = "Bytes")]
    pub signature: [u8; 64],
}

//Main device struct - handles proof generation/verification and aggregation
pub struct IoTDevice {
    pub id: u32,
    pub frost_key: frost::keys::KeyPackage,
    pub group_pub: frost::keys::PublicKeyPackage,
    pub sig_key: SigningKey,
    pub peer_keys: HashMap<u32, ed_vf>,
    pub valid_participant_ids: HashSet<u32>,
    pub verified_ciphertexts: HashMap<u32, VerifiedCiphertext>,
    partials: HashMap<u32, VerifiedPartial>,
    agg_c1: Option<RistrettoPoint>,
    agg_c2: Option<RistrettoPoint>,
    halo2_setup: Halo2Setup,
    threshold: usize,
    rates: HashMap<u32, (u64, u32)>,
    seen_nonces: HashMap<u32, HashSet<[u8; 32]>>, //tracks v_commit (32-byte Fr repr)
}
impl IoTDevice {
    pub fn new(id: u32, threshold: usize, frost_key: frost::keys::KeyPackage, group_pub: frost::keys::PublicKeyPackage, peer_keys: HashMap<u32, ed_vf>, halo2_setup: Halo2Setup, signing_key: Option<SigningKey>) -> Result<Self, AggError> {
        //Basic checks
        if id == 0 { return Err(AggError::CryptoError("Device ID cannot be zero (Lagrange requirement)".into())); }
        if threshold == 0 { return Err(AggError::CryptoError("Threshold must be at least 1".into())); }
        if !peer_keys.is_empty() {
            let total_devices = peer_keys.len() + 1;
            if threshold > total_devices { return Err(AggError::CryptoError( format!("Threshold {} exceeds total devices {}", threshold, total_devices) )); }
            if total_devices > MAX_DEVICES { return Err(AggError::CryptoError( format!("Too many devices: {} > {}", total_devices, MAX_DEVICES) )); }
        }
        //Initialization
        let mut valid_participant_ids = HashSet::new();
        valid_participant_ids.insert(id);
        for peer_id in peer_keys.keys() { valid_participant_ids.insert(*peer_id); }
        let sig_key = signing_key.unwrap_or_else(|| SigningKey::generate(&mut OsRng));
        Ok(Self {
            id, threshold, frost_key, group_pub, peer_keys,
            sig_key, valid_participant_ids, halo2_setup,
            verified_ciphertexts: HashMap::new(), partials: HashMap::new(),
            agg_c1: None, agg_c2: None,
            rates: HashMap::new(),
            seen_nonces: HashMap::new(),
        })
    }
    //Generate proof for our state (0 or 1)
    pub fn generate_proof(&self, state: u8) -> Result<DeviceProof, AggError> {
        if state > 1 { return Err(AggError::CryptoError("State must be 0/1".into())); }
        let ts = timestamp();
        let mut r = Scalar::random(&mut OsRng);
        let g = RISTRETTO_BASEPOINT_POINT;
        let h = frost_to_point(&self.group_pub.verifying_key())?;
        let s_scalar = Scalar::from(state as u64);
        let c1 = g * r;
        let c2 = g * s_scalar + h * r;
        //epsilon = phi(c2) via XOR-fold
        let pedersen_bytes: [u8; 32] = c2.compress().to_bytes();
        let epsilon = point_to_fr(&pedersen_bytes);
        //Sample blinding factor b
        let mut b_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut b_bytes);
        b_bytes[31] &= 0x3F; //clamp high bits so value is canonically < r
        let blinding = Halo2Fr::from_repr(b_bytes).unwrap_or(Halo2Fr::ZERO);
        //Sample Schnorr nonce v; v_f = v embedded in Fr (no reduction since q < r)
        let v = Scalar::random(&mut OsRng);
        let v_f = scalar_to_fr(&v);
        //Compute V_commit BEFORE Fiat-Shamir
        let s_fr = Halo2Fr::from(state as u64);
        let v_commit_fr = mimc_hash(v_f, s_fr, epsilon, blinding);
        let v_commit_bytes32: [u8; 32] = v_commit_fr.to_repr();
        //For Schnorr transcript we only need 16 bytes for domain separation
        let mut v_commit_16 = [0u8; 16];
        v_commit_16.copy_from_slice(&v_commit_bytes32[..16]);
        //Schnorr proof with V_commit in transcript
        let (eg_proof, challenge) = ElGamalProof::prove_with_nonce(state, &r, &v, &c1, &c2, &h, self.id, ts, &v_commit_16);
        let resp_f = scalar_to_fr(&eg_proof.resp_state.0);
        let c_f    = scalar_to_fr(&challenge);
        //Build circuit and prove
        let circuit = BinaryCircuit {
            state: Value::known(s_fr),
            v_f: Value::known(v_f),
            epsilon: Value::known(epsilon),
            blinding: Value::known(blinding),
        };
        let instance_v_commit = vec![v_commit_fr];
        let instance_eps = vec![epsilon];
        let instance_resp_f = vec![resp_f];
        let instance_c_f = vec![c_f];
        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(&self.halo2_setup.params,&self.halo2_setup.pk,&[circuit],&[&[&instance_v_commit[..], &instance_eps[..], &instance_resp_f[..], &instance_c_f[..]]],OsRng,&mut transcript,).map_err(|e| { r.zeroize(); AggError::CryptoError(format!("Halo2 failed: {:?}", e)) })?;
        r.zeroize();
        let resp_f_bytes: [u8; 32] = resp_f.to_repr();
        let c_f_bytes:    [u8; 32] = c_f.to_repr();
        //Sign
        let mut sig_data = Vec::new();
        sig_data.extend_from_slice(&ts.to_le_bytes());
        sig_data.extend_from_slice(&self.id.to_le_bytes());
        sig_data.extend_from_slice(c1.compress().as_bytes());
        sig_data.extend_from_slice(c2.compress().as_bytes());
        sig_data.extend_from_slice(eg_proof.pedersen_commit.0.as_bytes());
        sig_data.extend_from_slice(eg_proof.commit_r.0.as_bytes());
        sig_data.extend_from_slice(eg_proof.commit_s.0.as_bytes());
        sig_data.extend_from_slice(eg_proof.commit_p.0.as_bytes());
        sig_data.extend_from_slice(&eg_proof.resp_r.0.to_bytes());
        sig_data.extend_from_slice(&eg_proof.resp_state.0.to_bytes());
        sig_data.extend_from_slice(&v_commit_bytes32);
        sig_data.extend_from_slice(&resp_f_bytes);
        sig_data.extend_from_slice(&c_f_bytes);
        let signature = self.sig_key.sign(&sig_data).to_bytes();
        Ok(DeviceProof {
            device_id: self.id, timestamp: ts,
            elgamal_c1: c1.compress().into(),
            elgamal_c2: c2.compress().into(),
            elgamal_proof: eg_proof,
            v_commit: v_commit_bytes32,
            resp_f: resp_f_bytes,
            c_f: c_f_bytes,
            halo2_proof: transcript.finalize(),
            signature,
        })
    }
    //Receive and verify a proof from a peer
    pub fn receive_proof(&mut self, p: DeviceProof) -> Result<(), AggError> {
        if self.verified_ciphertexts.len() >= MAX_STORED_PROOFS {
            self.cleanup();
            if self.verified_ciphertexts.len() >= MAX_STORED_PROOFS {
                return Err(AggError::RateLimited);
            }
        }
        check_rate(p.device_id, &mut self.rates)?;
        let now = timestamp();
        let adjusted_now = now + MAX_CLOCK_SKEW;
        if p.timestamp > adjusted_now {
            return Err(AggError::InvalidProof("Timestamp too far in future".into()));
        }
        if p.timestamp + PROOF_EXPIRY < now.saturating_sub(MAX_CLOCK_SKEW) {
            return Err(AggError::ExpiredProof);
        }
        let device_nonces = self.seen_nonces.entry(p.device_id).or_insert_with(HashSet::new);
        if device_nonces.len() >= MAX_NONCES_PER_DEVICE {
            return Err(AggError::RateLimited);
        }
        if !device_nonces.insert(p.v_commit) {
            return Err(AggError::InvalidProof("Nonce already used".into()));
        }
        if p.halo2_proof.len() > MAX_PROOF_SIZE {
            return Err(AggError::InvalidProof("Too big".into()));
        }
        if self.verified_ciphertexts.contains_key(&p.device_id) {
            return Err(AggError::InvalidProof("Duplicate".into()));
        }
        //Signature
        let pk = self.peer_keys.get(&p.device_id).ok_or(AggError::InvalidProof("Unknown device".into()))?;
        let mut sig_data = Vec::new();
        sig_data.extend_from_slice(&p.timestamp.to_le_bytes());
        sig_data.extend_from_slice(&p.device_id.to_le_bytes());
        sig_data.extend_from_slice(p.elgamal_c1.0.as_bytes());
        sig_data.extend_from_slice(p.elgamal_c2.0.as_bytes());
        sig_data.extend_from_slice(p.elgamal_proof.pedersen_commit.0.as_bytes());
        sig_data.extend_from_slice(p.elgamal_proof.commit_r.0.as_bytes());
        sig_data.extend_from_slice(p.elgamal_proof.commit_s.0.as_bytes());
        sig_data.extend_from_slice(p.elgamal_proof.commit_p.0.as_bytes());
        sig_data.extend_from_slice(&p.elgamal_proof.resp_r.0.to_bytes());
        sig_data.extend_from_slice(&p.elgamal_proof.resp_state.0.to_bytes());
        sig_data.extend_from_slice(&p.v_commit);
        sig_data.extend_from_slice(&p.resp_f);
        sig_data.extend_from_slice(&p.c_f);
        let sig = Signature::try_from(&p.signature[..]).map_err(|_| AggError::InvalidProof("bad sig".into()))?;
        pk.verify(&sig_data, &sig).map_err(|_| AggError::InvalidProof("sig verify failed".into()))?;
        //ElGamal Schnorr (uses low 16 bytes of v_commit as transcript binding)
        let mut v_commit_16 = [0u8; 16];
        v_commit_16.copy_from_slice(&p.v_commit[..16]);
        let c1 = p.elgamal_c1.0.decompress().ok_or(AggError::InvalidProof("bad c1".into()))?;
        let c2 = p.elgamal_c2.0.decompress().ok_or(AggError::InvalidProof("bad c2".into()))?;
        let h = frost_to_point(&self.group_pub.verifying_key())?;
        if !p.elgamal_proof.verify(&c1, &c2, &h, p.device_id, p.timestamp, &v_commit_16) {
            return Err(AggError::InvalidProof("Schnorr failed".into()));
        }
        //Halo2 verify
        let v_commit_fr = Halo2Fr::from_repr(p.v_commit).unwrap_or(Halo2Fr::ZERO);
        let resp_f_fr = Halo2Fr::from_repr(p.resp_f).unwrap_or(Halo2Fr::ZERO);
        let c_f_fr = Halo2Fr::from_repr(p.c_f).unwrap_or(Halo2Fr::ZERO);
        let pedersen_bytes: [u8; 32] = c2.compress().to_bytes();
        let epsilon_fr = point_to_fr(&pedersen_bytes);
        let instance_v_commit = vec![v_commit_fr];
        let instance_eps = vec![epsilon_fr];
        let instance_resp_f = vec![resp_f_fr];
        let instance_c_f = vec![c_f_fr];
        let strategy = SingleStrategy::new(&self.halo2_setup.params);
        let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&p.halo2_proof[..]);
        verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<'_, Bn256>, _, _, _>(&self.halo2_setup.params,&self.halo2_setup.vk,strategy,&[&[&instance_v_commit[..], &instance_eps[..], &instance_resp_f[..], &instance_c_f[..]]],&mut transcript,).map_err(|_| AggError::InvalidProof("Halo2 verify failed".into()))?;
        self.verified_ciphertexts.insert(p.device_id, VerifiedCiphertext {timestamp: p.timestamp, c1, c2});
        Ok(())
    }
    pub fn cleanup(&mut self) {
        let cutoff = timestamp().saturating_sub(PROOF_EXPIRY);
        let expired_devices: HashSet<u32> = self.verified_ciphertexts.iter().filter(|(_, vc)| vc.timestamp <= cutoff).map(|(id, _)| *id).collect();
        self.verified_ciphertexts.retain(|_, vc| vc.timestamp > cutoff);
        self.partials.retain(|_, p| p.timestamp > cutoff);
        for device_id in expired_devices { self.seen_nonces.remove(&device_id); }
    }
    fn recompute(&mut self) {
        if self.verified_ciphertexts.is_empty() { self.agg_c1 = None; self.agg_c2 = None; } else {
            let (mut c1, mut c2) = (RistrettoPoint::identity(), RistrettoPoint::identity());
            for vc in self.verified_ciphertexts.values() {
                c1+=vc.c1;
                c2+=vc.c2;
            }
            self.agg_c1 = Some(c1);
            self.agg_c2 = Some(c2);
        }
    }

    pub fn generate_partial_decryption(&mut self) -> Result<PartialDecryption, AggError> {
        self.recompute();
        let verified_vec: Vec<VerifiedCiphertext> = self.verified_ciphertexts.values().cloned().collect();
        let partial = crate::common::generate_partial_decryption(self.id,&self.frost_key,&self.group_pub,&self.sig_key,&verified_vec)?;
        Ok(partial)
    }
    pub fn receive_partial(&mut self, partial: PartialDecryption) -> Result<(), AggError> {
        let verified_vec: Vec<VerifiedCiphertext> = self.verified_ciphertexts.values().cloned().collect();
        let mut verified_partials_vec: Vec<VerifiedPartial> = self.partials.values().cloned().collect();
        let verified = crate::common::receive_partial(partial,&self.peer_keys,&self.group_pub,&verified_vec,&mut verified_partials_vec,&mut self.rates,)?;
        self.partials.insert(verified.device_id, verified);
        Ok(())
    }
    pub fn compute_aggregate(&mut self) -> Result<(usize, usize), AggError> {
        self.recompute();
        if self.partials.len() < self.threshold { return Err(AggError::ThresholdNotMet); }
        let verified_vec: Vec<VerifiedCiphertext> = self.verified_ciphertexts.values().cloned().collect();
        let verified_partials_vec: Vec<VerifiedPartial> = self.partials.values().cloned().collect();
        crate::common::compute_aggregate(self.threshold,&verified_vec,&verified_partials_vec)
    }
}