This repository is associated with zk-DEAP paper.

It is structured as follows: 

    zk-DEAP
    |
    │   .gitignore
    │   Cargo.lock: Defines libraries, dependencies and binaries
    |
    ├───src
    |   |
    │   │   lib.rs
    │   ├───common: Common methods shared across all three
    │   │       mod.rs
    │   ├───bulletproof: Bulletproof specific methods
    │   │       mod.rs
    │   ├───snark: SNARK specific methods
    │   │       mod.rs
    │   ├───stark: STARK specific methods
    │   │        mod.rs
    |   |
    │   └───bin
    │       ├───azureTestingScripts: Scripts used in Azure tests
    │       │       fixture_server.rs
    │       │       snark_setup.rs
    │       │       test_harness.rs
    │       ├───examples: Examples on how to use the zk-DEAP libraries
    │       │       bullet_example.rs
    │       │       snark_examples.rs
    │       │       stark_examples.rs
    │       ├───security_tests: Security tests on the library (common methods and one for each implementation)
    │       │       bullet_penTest.rs
    │       │       snark_penTest.rs
    │       │       stark_penTest.rs
    │       │       common_penTest.rs
    │       └───utils: Helper script to obtain the zk-SNARK public parameters
    |               param_gen.rs
    |
    ├───azureVMLogs: Raw data from the Azure tests
    │       VM3.jsonl
    |       ...
    │       VM8.jsonl
    │       vmLogAnalysis.py: Python script to interpret and analyze the logs
    |
    └───trusted_setup: Example KZG BN254_8 to use
            kzg_bn254_8.params