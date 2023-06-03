using ProgressMeter

# This file requires these global variables to be set:
# - MINIBATCH_SZ :: Int
# - EMBEDDING_SZ :: Int
# - STRIDE :: Int

const AMINO_ACID_TYPE = UInt8
const CATEGORY_TYPE = UInt8
const AMINO_ACID_BLANK = 25

struct PhANNsDataset
    cats_n :: Int64
    cats_lookup :: Array{Int64}
    cats :: Array{CATEGORY_TYPE}
    lens :: Array{UInt16}
    seqs :: Array{Array{AMINO_ACID_TYPE}}
end

function PhANNsDataset(cats, lens, seqs)
    cats_n = length(unique(cats))
    cats_lookup = [1]

    last_i = 1
    last_val = cats[1]
    for i=2:length(cats)
        if last_val != cats[i]
            push!(cats_lookup, i)
            last_val = cats[i]
        end
    end
    push!(cats_lookup, length(cats))

    PhANNsDataset(cats_n, cats_lookup, cats, lens, seqs)
end

include("preprocess_fasta.jl")

# statically allocated minibatch buffers
minibatch_x = AMINO_ACID_TYPE.(zeros(STRIDE, MINIBATCH_SZ))
minibatch_x_mangled = AMINO_ACID_TYPE.(zeros(STRIDE, MINIBATCH_SZ))
minibatch_x_mangled_p = Float32.(zeros(STRIDE, MINIBATCH_SZ))
minibatch_y = CATEGORY_TYPE.(zeros(MINIBATCH_SZ))

function apply_labels(data)
    res = zeros(length(data.cats), 11)
    @showprogress for i=1:length(data.cats)
        rebase = Chain(gpu(PositionalEncoder(EMBEDDING_SZ, length(data.seqs[i]))), base[2])
        g = sum(classify(rebase(embeddings(gpu(data.seqs[i])))), dims=2)
	      res[i,:] = cpu(g / sum(g))
    end
    return res
end

"Convert an ORF into a batch. Returns a matrix with all STRIDE (current $STRIDE) length substrings."
function as_batch(s)
    n = max(length(s) - (STRIDE - 1), 1)

    batch = AMINO_ACID_TYPE.(zeros(STRIDE, n))
    for i=1:size(batch, 2)
        for j=1:size(batch, 1)
            if length(s) >= j + i
                batch[j, i] = s[j + i]
            else
                batch[j, i] = AMINO_ACID_BLANK
            end
        end
    end
    return batch
end

"Convert an ORF into a batch. Returns a matrix with all STRIDE (current $STRIDE) length substrings."
function apply_labels_to_file(file, fn)
    seqs = sequences_in_file(file)
    apply_label_to_seqs(seqs, fn)
end

function apply_label_to_seqs(seqs, fn)
    res = zeros(length(seqs), 11)
    @showprogress for i=1:length(seqs)
        g = sum(fn(gpu(as_batch(seqs[i]))), dims=[2, 3])
	      res[i,:] = cpu(g ./ sum(g))
    end
    return res
end

function load_minibatch!(data :: PhANNsDataset, minibatch_x :: Matrix{AMINO_ACID_TYPE}, minibatch_x_mangled :: Matrix{AMINO_ACID_TYPE}, minibatch_x_mangled_p :: Matrix{Float32}, minibatch_y :: Array{AMINO_ACID_TYPE}, p :: Float64)
    @assert size(minibatch_x_mangled, 2) == length(minibatch_y)
    @assert size(minibatch_x, 2) == length(minibatch_y)

    for i=1:size(minibatch_x, 2)
        cat = rand(1:data.cats_n)
        di = rand((data.cats_lookup[cat]):(data.cats_lookup[cat+1]))
        offset = (data.lens[di] > STRIDE) ? rand(1:(data.lens[di] - STRIDE)) : 0
        for j=1:size(minibatch_x, 1)
            if 0 < (j + offset) <= data.lens[di]
                @inbounds minibatch_x[j,i] = data.seqs[di][j + offset]
            else
                @inbounds minibatch_x[j,i] = AMINO_ACID_BLANK
            end

            if 0 < (j + offset) <= data.lens[di]
                if rand() < p
                    @inbounds minibatch_x_mangled[j,i] = rand(1:25) # random amino acid
                    @inbounds minibatch_x_mangled_p[j,i] = 1.0
                else
                    @inbounds minibatch_x_mangled[j,i] = data.seqs[di][j + offset]
                    @inbounds minibatch_x_mangled_p[j,i] = 0.0
                end
            else
                @inbounds minibatch_x_mangled[j,i] = AMINO_ACID_BLANK
                @inbounds minibatch_x_mangled_p[j,i] = 0.0
            end
        end
        minibatch_y[i] = cat
    end
end

load_minibatch!(minibatch_x :: Matrix{AMINO_ACID_TYPE}, minibatch_y :: Array{AMINO_ACID_TYPE}, p) =
    load_mangled_minibatch!(phanns_data, minibatch_x, minibatch_x_mangled, minibatch_x_mangled_p, minibatch_y, p)
load_minibatch!(p) = load_minibatch!(phanns_data, minibatch_x, minibatch_x_mangled, minibatch_x_mangled_p, minibatch_y, p)
load_minibatch_fn(p) = () -> load_minibatch!(phanns_data, minibatch_x, minibatch_x_mangled, minibatch_x_mangled_p, minibatch_y, p)
