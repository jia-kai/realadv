using JuMP
using MIPVerify
using Gurobi
using Memento
using MAT

param_dict = ARGS[1] |> matread
inp_size = parse(Int, ARGS[2])
inp_chl = parse(Int, ARGS[3])
time_limit = parse(Int, ARGS[4])
nr_threads = parse(Int, ARGS[5])

c1_size = inp_size * inp_size ÷ 4 * 16
c2_size = inp_size * inp_size ÷ 16 * 32
c3_size = 100

# code copied from verify_MNIST.jl from MadryLab/relu_stable
fc1 = get_matrix_params(param_dict, "fc1",
                        (inp_size*inp_size*inp_chl, c1_size))
if haskey(param_dict, "fc1/mask")
    m1 = MaskedReLU(squeeze(param_dict["fc1/mask"], 1), interval_arithmetic)
else
    m1 = ReLU(interval_arithmetic)
end
fc2 = get_matrix_params(param_dict, "fc2", (c1_size, c2_size))
if haskey(param_dict, "fc2/mask")
    m2 = MaskedReLU(squeeze(param_dict["fc2/mask"], 1))
else
    m2 = ReLU()
end
fc3 = get_matrix_params(param_dict, "fc3", (c2_size, c3_size))
if haskey(param_dict, "fc3/mask")
    m3 = MaskedReLU(squeeze(param_dict["fc3/mask"], 1))
else
    m3 = ReLU()
end
softmax = get_matrix_params(param_dict, "softmax", (c3_size, 10))

nnparams = Sequential(
    [Flatten(4), fc1, m1, fc2, m2, fc3, m3, softmax],
    "$(model_name)"
)

function extract_results_for_save(d::Dict)::Dict
    m = d[:Model]
    r = Dict()
    r["SolveTime"] = d[:SolveTime]
    r["ObjectiveBound"] = getobjbound(m)
    r["ObjectiveValue"] = getobjectivevalue(m)
    r["TargetIndexes"] = d[:TargetIndexes]
    r["SolveStatus"] = string(d[:SolveStatus])
    r["PredictedIndex"] = d[:PredictedIndex]
    r["TighteningApproach"] = d[:TighteningApproach]
    r["TotalTime"] = d[:TotalTime]
    if !isnan(r["ObjectiveValue"])
        r["PerturbationValue"] = d[:Perturbation] |> getvalue
        r["PerturbedInputValue"] = d[:PerturbedInput] |> getvalue
    end
    return r
end

gurobi_env = Gurobi.Env()
setparam!(gurobi_env, "Threads", nr_threads)

while true
    io_file_name = readline()
    label = parse(Int, readline()) + 1
    eps = parse(Float64, readline())
    tolerance = parse(Float32, readline())

    println("new task: ", io_file_name)

    input_data = matread(io_file_name)["img"]
    target_indices = Int[]
    let x = 1
        while x <= 10
            if x != label
                push!(target_indices, x)
            end
            x += 1
        end
    end
    adv_result = MIPVerify.find_adversarial_example(
        nnparams,
        input_data,
        target_indices,
        GurobiSolver(gurobi_env, TimeLimit=time_limit),
        pp=MIPVerify.LInfNormBoundedPerturbationFamily(eps),
        norm_order=Inf,
        rebuild=false,
        cache_model=false,
        tightening_algorithm=lp,
        tightening_solver=GurobiSolver(gurobi_env, TimeLimit=10,
                                       OutputFlag=0),
        adversarial_example_objective=MIPVerify.worst_with_constraint,
        tolerance=tolerance,
    )
    save_d = extract_results_for_save(adv_result)
    matwrite(io_file_name, save_d)
    Base.Filesystem.touch(io_file_name * ".done")
end

