using Transformers, Flux, Statistics, ProgressMeter, BSON

"Simple wrapper to allow use of the attention heads from the Transformers package, disreguarding tokenization, positional encoding, and other features of that package."
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
