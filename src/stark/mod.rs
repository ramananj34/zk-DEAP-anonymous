use serde::{Deserialize, Serialize}; //Serialize/deserialize
use serde_with::{serde_as, Bytes};//Serialize/deserialize
use curve25519_dalek_ng::{ristretto::*, scalar::Scalar, traits::Identity, constants::RISTRETTO_BASEPOINT_POINT}; //My curve
use winterfell::{Air, AirContext, Assertion, EvaluationFrame, ProofOptions, Prover, Proof,TraceInfo, TransitionConstraintDegree,crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},math::{fields::f128::BaseElement, FieldElement, ToElements, StarkField},matrix::ColMatrix, TraceTable, verify, AcceptableOptions,ConstraintCompositionCoefficients, DefaultConstraintEvaluator, DefaultTraceLde,PartitionOptions, StarkDomain, TracePolyTable, AuxRandElements}; //Winterfell
use winter_utils::Serializable; //Winterfell
use merlin::Transcript; //Transcript
use frost_ristretto255 as frost; //FROST
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier}; //Digital signatures
use zeroize::Zeroize; //Safe data deletion
use rand::rngs::OsRng; //Random
use rand::RngCore; //Random
use std::collections::{HashMap, HashSet}; //Data structures

use crate::common::{SerCompressed, SerScalar, AggError, VerifiedCiphertext, VerifiedPartial, PartialDecryption, PROTOCOL_VERSION, MAX_DEVICES, MAX_STORED_PROOFS, MAX_CLOCK_SKEW,MAX_NONCES_PER_DEVICE, PROOF_EXPIRY, timestamp, frost_to_point, check_rate};

const MAX_PROOF_SIZE: usize = 65536; //prevent DoS attacks
const TRACE_LENGTH: usize = 64;
const MIMC_ROUNDS: usize = TRACE_LENGTH - 1; //63 rounds, one per transition

//Public MiMC round constants (nothing-up-my-sleeve, from BLAKE3)
fn mimc_round_constants() -> [BaseElement; TRACE_LENGTH] {
    let mut rc = [BaseElement::ZERO; TRACE_LENGTH];
    for i in 0..TRACE_LENGTH {
        let label = format!("zk-DEAP-MiMC-rc-{}", i);
        let h = blake3::hash(label.as_bytes());
        let bytes = h.as_bytes();
        let mut val: u128 = 0;
        for j in 0..16 { val |= (bytes[j] as u128) << (j * 8); }
        rc[i] = BaseElement::new(val);
    }
    rc
}

//MiMC_63 hash: x_{i+1} = (x_i + s + eps + b + rc_i)^5 for i in 0..63
fn mimc_hash(v_f: BaseElement, s: BaseElement, eps: BaseElement, b: BaseElement) -> BaseElement {
    let rc = mimc_round_constants();
    let mut x = v_f;
    for i in 0..MIMC_ROUNDS {
        let sum = x + s + eps + b + rc[i];
        let sq = sum * sum;
        x = sq * sq * sum; // x^5
    }
    x
}

//Encode a 32-byte compressed Ristretto point as F_p via XOR-fold
pub fn point_to_field_element(bytes: &[u8; 32]) -> BaseElement {
    let mut val: u128 = 0;
    for i in 0..16 { val |= ((bytes[i] ^ bytes[i + 16]) as u128) << (i * 8); }
    BaseElement::new(val)
}

//Reduce Ristretto scalar (Z_q, q ~ 2^252) to F_p where p = 2^128 - 45*2^40 + 1. Uses s = s_lo + s_hi * 2^128 and 2^128 mod p = 45*2^40 - 1.
fn scalar_to_field_element(s: &Scalar) -> BaseElement {
    let bytes = s.to_bytes();
    let mut s_lo: u128 = 0;
    let mut s_hi: u128 = 0;
    for i in 0..16 {
        s_lo |= (bytes[i] as u128) << (i * 8);
        s_hi |= (bytes[i + 16] as u128) << (i * 8);
    }
    let lo = BaseElement::new(s_lo);
    let hi = BaseElement::new(s_hi);
    let two_128_mod_p = BaseElement::new(45u128 * (1u128 << 40) - 1);
    lo + hi * two_128_mod_p
}

//q mod p, precomputed via Scalar arithmetic (q - 1 mod p, then add 1)
fn q_mod_p() -> BaseElement {
    scalar_to_field_element(&(-Scalar::one())) + BaseElement::ONE
}


//STARK AIR definition - enforces binary constraint
#[derive(Clone, Debug)]
pub struct BinaryPublicInputs {
    pub v_commit: BaseElement,
    pub epsilon: BaseElement,
    pub resp_f: BaseElement,
    pub c_f: BaseElement,
}
//Convert to linear array for STARK framework
impl ToElements<BaseElement> for BinaryPublicInputs { fn to_elements(&self) -> Vec<BaseElement> { vec![self.v_commit, self.epsilon, self.resp_f, self.c_f] } }
//AIR Algebraic Intermediate Representation. This defines the constraint that state * (state - 1) = 0
#[allow(dead_code)]
#[derive(Clone)]
pub struct BinaryAir {
    context: AirContext<BaseElement>,
    v_commit: BaseElement,
    epsilon: BaseElement,
    resp_f: BaseElement,
    c_f: BaseElement,
    q_mod_p: BaseElement,
}
impl Air for BinaryAir {
    type BaseField = BaseElement;
    type PublicInputs = BinaryPublicInputs;
    type GkrProof = ();
    type GkrVerifier = ();
    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, opts: ProofOptions) -> Self {
        assert_eq!(6, trace_info.width());
        let degrees = vec![
            TransitionConstraintDegree::new(2), //C0 binary s
            TransitionConstraintDegree::with_cycles(5, vec![TRACE_LENGTH]), //C1 MiMC (periodic rc)
            TransitionConstraintDegree::new(1), //C2 s constancy
            TransitionConstraintDegree::new(1), //C3 eps constancy
            TransitionConstraintDegree::new(1), //C4 b constancy
            TransitionConstraintDegree::new(1), //C5 schnorr aux
            TransitionConstraintDegree::new(2), //C6 omega binary
            TransitionConstraintDegree::new(1), //C7 omega constancy
        ];
        Self {
            context: AirContext::new(trace_info, degrees, 3, opts),
            v_commit: pub_inputs.v_commit,
            epsilon: pub_inputs.epsilon,
            resp_f: pub_inputs.resp_f,
            c_f: pub_inputs.c_f,
            q_mod_p: q_mod_p(),
        }
    }
    fn context(&self) -> &AirContext<Self::BaseField> { &self.context }
    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(&self, frame: &EvaluationFrame<E>, periodic_values: &[E], result: &mut [E]) {
        let cur = frame.current();
        let nxt = frame.next();
        let rc_i = periodic_values[0];
        let c_f = E::from(self.c_f);
        let q_mp = E::from(self.q_mod_p);
        let s = cur[0];
        let m = cur[1];
        let eps = cur[2];
        let b = cur[3];
        let aux = cur[4];
        let omega = cur[5];
        //C0: binary s
        result[0] = s * (s - E::ONE);
        //C1: MiMC round  m_{i+1} = (m_i + s + eps + b + rc_i)^5
        let sum = m + s + eps + b + rc_i;
        let sq = sum * sum;
        let pow5 = sq * sq * sum;
        result[1] = nxt[1] - pow5;
        //C2..C4: constancy
        result[2] = nxt[0] - s;
        result[3] = nxt[2] - eps;
        result[4] = nxt[3] - b;
        //C5: schnorr aux  T[4,i] = T[1,i] + c_f * T[0,i] - omega * q_mod_p
        result[5] = aux - m - c_f * s + omega * q_mp;
        //C6: omega binary
        result[6] = omega * (omega - E::ONE);
        //C7: omega constancy
        result[7] = nxt[5] - omega;
    }
    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        vec![Assertion::single(1, TRACE_LENGTH - 1, self.v_commit), Assertion::single(2, 0, self.epsilon), Assertion::single(4, 0, self.resp_f) ]
    }
    fn get_periodic_column_values(&self) -> Vec<Vec<Self::BaseField>> {
        vec![mimc_round_constants().to_vec()]
    }
}
//STARK prover implementation
#[derive(Clone)]
pub struct BinaryProver {
    pub options: ProofOptions,
    last_v_commit: std::cell::Cell<Option<BaseElement>>,
    last_epsilon: std::cell::Cell<Option<BaseElement>>,
    last_resp_f: std::cell::Cell<Option<BaseElement>>,
    last_c_f: std::cell::Cell<Option<BaseElement>>,
}
impl BinaryProver {
    pub fn new() -> Self {
        Self {
            options: ProofOptions::new(40, 16, 20, winterfell::FieldExtension::None, 8, 63),
            last_v_commit: std::cell::Cell::new(None),
            last_epsilon: std::cell::Cell::new(None),
            last_resp_f: std::cell::Cell::new(None),
            last_c_f: std::cell::Cell::new(None),
        }
    }
    pub fn build_trace(&self, state: u8, v_f: BaseElement, epsilon: BaseElement, blinding: BaseElement, resp_f: BaseElement, c_f: BaseElement, omega: BaseElement) -> TraceTable<BaseElement> {
        let s_elem = BaseElement::new(state as u128);
        let rc = mimc_round_constants();
        let q_mp = q_mod_p();
        let mut mimc = vec![BaseElement::ZERO; TRACE_LENGTH];
        mimc[0] = v_f;
        for i in 0..MIMC_ROUNDS {
            let sum = mimc[i] + s_elem + epsilon + blinding + rc[i];
            let sq = sum * sum;
            mimc[i + 1] = sq * sq * sum;
        }
        let v_commit = mimc[TRACE_LENGTH - 1];
        self.last_v_commit.set(Some(v_commit));
        self.last_epsilon.set(Some(epsilon));
        self.last_resp_f.set(Some(resp_f));
        self.last_c_f.set(Some(c_f));
        let mut trace = TraceTable::new(6, TRACE_LENGTH);
        trace.fill(
            |row| {
                row[0] = s_elem;
                row[1] = v_f;
                row[2] = epsilon;
                row[3] = blinding;
                row[4] = v_f + c_f * s_elem - omega * q_mp;
                row[5] = omega;
            },
            |step, row| {
                let m_next = mimc[step + 1];
                row[0] = s_elem;
                row[1] = m_next;
                row[2] = epsilon;
                row[3] = blinding;
                row[4] = m_next + c_f * s_elem - omega * q_mp;
                row[5] = omega;
            },
        );
        trace
    }
}
//Prover biolerplate
impl Prover for BinaryProver {
    type BaseField = BaseElement;
    type Air = BinaryAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3_256<BaseElement>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type VC = MerkleTree<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> = DefaultConstraintEvaluator<'a, Self::Air, E>;
    fn get_pub_inputs(&self, _: &Self::Trace) -> BinaryPublicInputs {
        BinaryPublicInputs {
            v_commit: self.last_v_commit.get().unwrap(),
            epsilon: self.last_epsilon.get().unwrap(),
            resp_f: self.last_resp_f.get().unwrap(),
            c_f: self.last_c_f.get().unwrap(),
        }
    }
    fn options(&self) -> &ProofOptions { &self.options }
    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(&self, info: &TraceInfo, main: &ColMatrix<Self::BaseField>, domain: &StarkDomain<Self::BaseField>, part: PartitionOptions) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(info, main, domain, part)
    }
    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(&self, air: &'a Self::Air, aux: Option<AuxRandElements<E>>,coeffs: ConstraintCompositionCoefficients<E>) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux, coeffs)
    }
}

//ElGamal correctness proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElGamalProof {
    pub commit_r: SerCompressed,
    pub commit_s: SerCompressed,
    pub commit_p: SerCompressed,
    pub resp_r: SerScalar,
    pub resp_state: SerScalar,
    pub pedersen_commit: SerCompressed
}
impl ElGamalProof {
    //Schnorr prove with caller-supplied nonce v. V_commit is bound into the transcript before challenge derivation. Returns (proof, challenge).
    pub fn prove_with_nonce(state: u8, r: &Scalar, v: &Scalar, c1: &RistrettoPoint, c2: &RistrettoPoint, h: &RistrettoPoint, dev_id: u32, ts: u64, v_commit_bytes: &[u8; 16] ) -> (Self, Scalar) {
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

    pub fn verify(&self, c1: &RistrettoPoint, c2: &RistrettoPoint, h: &RistrettoPoint, dev_id: u32, ts: u64, v_commit_bytes: &[u8; 16] ) -> bool {
        let g = RISTRETTO_BASEPOINT_POINT;
        let (Some(cr), Some(cs), Some(cp), Some(pc)) = (
            self.commit_r.0.decompress(),
            self.commit_s.0.decompress(),
            self.commit_p.0.decompress(),
            self.pedersen_commit.0.decompress(),
        ) else { return false };
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

//Device proof with STARK
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProof {
    pub device_id: u32,
    pub timestamp: u64,
    pub elgamal_c1: SerCompressed,
    pub elgamal_c2: SerCompressed,
    pub elgamal_proof: ElGamalProof,
    pub v_commit: [u8; 16],
    pub resp_f: [u8; 16],
    pub c_f: [u8; 16],
    pub omega: u8,
    pub stark_proof: Vec<u8>,
    #[serde_as(as = "Bytes")]
    pub signature: [u8; 64],
}

//Main device struct - handles proof generation/verification and aggregation
pub struct IoTDevice {
    pub id: u32,
    pub frost_key: frost::keys::KeyPackage,
    pub group_pub: frost::keys::PublicKeyPackage,
    pub sig_key: SigningKey,
    pub peer_keys: HashMap<u32, VerifyingKey>,
    pub valid_participant_ids: HashSet<u32>,
    pub verified_ciphertexts: HashMap<u32, VerifiedCiphertext>,
    partials: HashMap<u32, VerifiedPartial>,
    agg_c1: Option<RistrettoPoint>,
    agg_c2: Option<RistrettoPoint>,
    stark_prover: BinaryProver,
    threshold: usize,
    rates: HashMap<u32, (u64, u32)>,
    seen_nonces: HashMap<u32, HashSet<[u8; 16]>>,
}

impl IoTDevice {
    pub fn new(id: u32, threshold: usize, frost_key: frost::keys::KeyPackage, group_pub: frost::keys::PublicKeyPackage, peer_keys: HashMap<u32, VerifyingKey>, signing_key: Option<SigningKey>) -> Result<Self, AggError> {
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
            sig_key, valid_participant_ids,
            verified_ciphertexts: HashMap::new(), partials: HashMap::new(),
            agg_c1: None, agg_c2: None,
            stark_prover: BinaryProver::new(),
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
        let pedersen_bytes: [u8; 32] = c2.compress().to_bytes();
        let epsilon = point_to_field_element(&pedersen_bytes);
        let mut b_bytes = [0u8; 16];
        OsRng.fill_bytes(&mut b_bytes);
        let blinding = BaseElement::new(u128::from_le_bytes(b_bytes));
        let v = Scalar::random(&mut OsRng);
        let v_f = scalar_to_field_element(&v);
        let s_elem = BaseElement::new(state as u128);
        let v_commit_elem = mimc_hash(v_f, s_elem, epsilon, blinding);
        let v_commit_bytes: [u8; 16] = v_commit_elem.as_int().to_le_bytes();
        let (eg_proof, challenge) = ElGamalProof::prove_with_nonce(state, &r, &v, &c1, &c2, &h, self.id, ts, &v_commit_bytes);
        //omega: carry bit in {0,1}. For state=0 no addition happens so omega=0. For state=1, omega=1 iff v+c >= q as integers, equivalently (v+c) mod q < v in canonical LE order.
        let omega = if state == 1 {
            let sum = v + challenge;
            let sum_b = sum.to_bytes();
            let v_b = v.to_bytes();
            let mut wrap = false;
            for i in (0..32).rev() {
                if sum_b[i] < v_b[i] { wrap = true; break; }
                if sum_b[i] > v_b[i] { break; }
            }
            if wrap { BaseElement::ONE } else { BaseElement::ZERO }
        } else {
            BaseElement::ZERO
        };
        let resp_f = scalar_to_field_element(&eg_proof.resp_state.0);
        let c_f    = scalar_to_field_element(&challenge);
        let trace = self.stark_prover.build_trace(state, v_f, epsilon, blinding, resp_f, c_f, omega);
        let stark_proof = self.stark_prover.prove(trace).map_err(|e| {r.zeroize();AggError::CryptoError(format!("STARK failed: {:?}", e))})?;
        let mut stark_bytes = Vec::new();
        stark_proof.write_into(&mut stark_bytes);
        r.zeroize();
        let resp_f_bytes: [u8; 16] = resp_f.as_int().to_le_bytes();
        let c_f_bytes: [u8; 16] = c_f.as_int().to_le_bytes();
        let omega_byte: u8 = if omega == BaseElement::ONE { 1 } else { 0 };
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
        sig_data.extend_from_slice(&v_commit_bytes);
        sig_data.extend_from_slice(&resp_f_bytes);
        sig_data.extend_from_slice(&c_f_bytes);
        sig_data.push(omega_byte);
        let signature = self.sig_key.sign(&sig_data).to_bytes();
        Ok(DeviceProof {
            device_id: self.id, timestamp: ts,
            elgamal_c1: c1.compress().into(),
            elgamal_c2: c2.compress().into(),
            elgamal_proof: eg_proof,
            v_commit: v_commit_bytes,
            resp_f: resp_f_bytes,
            c_f: c_f_bytes,
            omega: omega_byte,
            stark_proof: stark_bytes,
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
        if p.timestamp > now + MAX_CLOCK_SKEW {
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
        if self.verified_ciphertexts.contains_key(&p.device_id) {
            return Err(AggError::InvalidProof("Duplicate".into()));
        }
        if p.stark_proof.len() > MAX_PROOF_SIZE {
            return Err(AggError::InvalidProof("Too big".into()));
        }
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
        sig_data.push(p.omega);
        let sig = Signature::try_from(&p.signature[..]).map_err(|_| AggError::InvalidProof("bad sig".into()))?;
        pk.verify(&sig_data, &sig).map_err(|_| AggError::InvalidProof("sig verify failed".into()))?;
        let c1 = p.elgamal_c1.0.decompress().ok_or(AggError::InvalidProof("bad c1".into()))?;
        let c2 = p.elgamal_c2.0.decompress().ok_or(AggError::InvalidProof("bad c2".into()))?;
        let h = frost_to_point(&self.group_pub.verifying_key())?;
        if !p.elgamal_proof.verify(&c1, &c2, &h, p.device_id, p.timestamp, &p.v_commit) {
            return Err(AggError::InvalidProof("Schnorr failed".into()));
        }
        let v_commit_elem = BaseElement::new(u128::from_le_bytes(p.v_commit));
        let resp_f_elem= BaseElement::new(u128::from_le_bytes(p.resp_f));
        let c_f_elem= BaseElement::new(u128::from_le_bytes(p.c_f));
        let pedersen_bytes: [u8; 32] = c2.compress().to_bytes();
        let epsilon_elem = point_to_field_element(&pedersen_bytes);
        let stark_proof = Proof::from_bytes(&p.stark_proof[..]).map_err(|_| AggError::InvalidProof("bad STARK format".into()))?;
        let pub_inputs = BinaryPublicInputs {
            v_commit: v_commit_elem,
            epsilon:  epsilon_elem,
            resp_f:   resp_f_elem,
            c_f:      c_f_elem,
        };
        let min_opts = AcceptableOptions::MinConjecturedSecurity(95);
        verify::<BinaryAir, Blake3_256<BaseElement>, DefaultRandomCoin<Blake3_256<BaseElement>>, MerkleTree<Blake3_256<BaseElement>>>(stark_proof, pub_inputs, &min_opts).map_err(|_| AggError::InvalidProof("STARK verify failed".into()))?;
        self.verified_ciphertexts.insert(p.device_id, VerifiedCiphertext {timestamp: p.timestamp, c1, c2});
        Ok(())
    }
    //Clean up old proofs and partials
    pub fn cleanup(&mut self) {
        let cutoff = timestamp().saturating_sub(PROOF_EXPIRY);
        let expired_devices: HashSet<u32> = self.verified_ciphertexts.iter().filter(|(_, vc)| vc.timestamp <= cutoff).map(|(id, _)| *id).collect();
        self.verified_ciphertexts.retain(|_, vc| vc.timestamp > cutoff);
        self.partials.retain(|_, p| p.timestamp > cutoff);
        for device_id in expired_devices { self.seen_nonces.remove(&device_id); }
    }
    //Recompute aggregate ciphertexts
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