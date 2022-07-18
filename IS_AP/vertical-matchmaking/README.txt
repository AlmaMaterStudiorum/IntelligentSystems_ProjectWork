HADA is an optimization engine for hardware dimensioning and algorithm configuration, designed as the vertical matchmaking layer of the European AI on-demand platform.

In the current version, the capabilities of HADA are limited to the Transprecision Computing use case, precisely to the Saxpy algorithm: the engine receives a collection of user-defined requirements on the run of the algorithm, and returns an optimal algorithm configuration and the least expensive platform, to execute the run while respecting the user needs. 

Theoretically, HADA is a Constraint Programming model made up of two components:
    1. An empirical, user-independent component: a dedicated predictive model for each pair (hw, target), used to estimate the performance of Saxpy, in terms of this target, while running on this hw platform;
    2. A declarative, user-dependent component: optimization criteria, user-defined constraints.

Practically, HADA consists of two python scripts: 
    1. build_model.py constructs the user-independent elements of HADA: it defines the basic variables and constraints, and embeds the required machine-learned modules into the model;
    2. solve_model.py complements and solves the basic HADA: it adds the objective and user-defined constraints on top of the pre-built basic model, hence it runs the solve and outputs an optimal solution.

Considerations on the chosen implementation design.
- Benefits: the embedding of the predictive models, which represents, computationally speaking, the most expensive phase, is performed offline. Once that the basic model is built, it can be arbitrarily used to deliver the optimal matching quickly and efficiently.
- Drawbacks: each predictive model needs to be embedded into HADA, regardless of whether the corresponding target is constrained by the user. This leads to a non-ignorable increase in the size of the final optimization model.

Future developments
1. Extend the capabilities of HADA to the full range of Transprecision Computing algorithms: BlackScholes, Convolution, Correlation, FWT, Saxpy.
2. Generalize the predictive models, by making them able to estimate the performance of an algorithm for multiple hw platforms and performance targets. This allows to reduce the number of empirical models that need to be embedded into HADA, hence to decrease the dimension of the final optimization engine. 


Usage Examples

1. Build basic HADA:
    
    python3 build_model.py \
    --data_folder <folder_containing_datasets_for_computing_upper_and_lower_bounds_of_variables> \
    --models_folder <folder_containing_predictive_models_to_embed>

2. Input optimization criteria and user-defined constraints, hence obtain an optimal solution: 

    python3 solve_model.py \
    --basic_model <lp_file_of_basic_HADA> \
    --cost_pc <usage_cost_for_pc_platform> \
    --cost_g100 <usage_cost_for_g100_platform> \
    --cost_vm <usage_cost_for_vm_platform> \
    --constraint_time <"<=" or "==" or ">=")> <right-hand_side> \
    --constraint_memory <type <"<=" or "==" or ">="> <right-hand_side> \ 
    --constraint_error <type <"<=" or "==" or ">="> <right-hand_side> 
