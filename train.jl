
MINIBATCH_SZ = 128
EMBEDDING_SZ = 64
STRIDE = 128
HEAD_N = 8
GLOBAL_WIDTH = EMBEDDING_SZ * STRIDE

h(x) = round(x * 100) / 100
hm(x) = h(mean(x))
id = identity

include("dataset.jl")

using Transformers, Flux, Statistics, ProgressMeter, BSON

struct TBlock{T}
    t :: T
end
Flux.@functor TBlock
(t :: TBlock)(x) = t.t(x, nothing)[1]
tb(layers_n) = TBlock(Transformer(Transformers.TransformerBlock, layers_n, HEAD_N, EMBEDDING_SZ, HEAD_N, EMBEDDING_SZ * 2))

struct PositionalEncoder
    d
end
Flux.@functor PositionalEncoder
Flux.params(pe :: PositionalEncoder) = Flux.params(())

function PositionalEncoder(h :: Integer, w :: Integer)
    d = zeros(Float32, h, w)
    for i=1:h
        for j=1:w
            if j % 2 == 0
                d[i, j] = sin((i-1)/10_000^(2*(j-1)/w))
            else
                d[i, j] = cos((i-1)/10_000^(2*(j-1)/w))
            end
        end
    end
    PositionalEncoder(d)
end
function (pe :: PositionalEncoder)(x)
   sz = size(pe.d)
   x .+ reshape(pe.d, (sz..., 1))
end


if !(@isdefined base)
    base = Chain(
        Embedding(25 => EMBEDDING_SZ),
        PositionalEncoder(EMBEDDING_SZ, STRIDE),
        tb(8)) |> gpu

    restore = Chain(
        Dense(EMBEDDING_SZ => 25),
        softmax) |> gpu

    classify = Chain(
        Dense(EMBEDDING_SZ => 11),
        softmax) |> gpu

    steps_taken = []
end

function train(train_br, train_bc, class_wt_fn, STEPS, p)
    push!(steps_taken, ("BRC", STEPS))
    losses = []
    fn = (Chain(base, restore), Chain(base, classify))
    optim = Flux.setup(Flux.Adam(0.001), fn)
    @showprogress for epoch in 1:STEPS
        load_minibatch!(p)
        gmbx_mangled = gpu(minibatch_x_mangled)
        gmbm_p = gpu(reshape(minibatch_x_mangled_p, (1, size(minibatch_x_mangled_p)...)))
        gmbx = gpu(minibatch_x)
        gmby = gpu(minibatch_y)
        loss, grads = Flux.withgradient(fn) do m
            _ret_loss = 0.0

            if train_br
                _ret_y_hat = m[1](embeddings(gmbx_mangled))
                _ret_loss = Flux.crossentropy(_ret_y_hat, gmbm_p .* Flux.onehotbatch(gmbx, 1:25))
            end

            _class_loss = 0.0
            if train_bc
                _class_y_hat = m[2](embeddings(gmbx))
                mby = Flux.onehotbatch(repeat(gmby, inner=STRIDE), 1:size(_class_y_hat, 1))
                mby = reshape(mby, size(_class_y_hat))
                _class_loss = Flux.crossentropy(_class_y_hat, mby)
            end

            sum(_ret_loss) + sum(_class_loss) * class_wt_fn(epoch)
        end
        Flux.update!(optim, fn, grads[1])
        push!(losses, loss)
    end
    mahd = min(100, Int64(div(STEPS, 2)))
    msg = "BRC NLL went from $(hm(losses[1:mahd])) to $(hm(losses[(end-mahd):end])): $(losses[end-5:end])"
    push!(steps_taken, ("BRC", STEPS, msg))
    println(msg)
    return losses
end
trainBRC(STEPS, p) = trainBRC(true, true, x -> 1, STEPS, p)
