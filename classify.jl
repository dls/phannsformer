doc = """classify.jl -- Classify ORFs using a pretrained model.

Usage:
  classify.jl --model 64-128 --input <file> --output <file>
  classify.jl -h | --help
  classify.jl --version

Options:
  -h --help          Show this screen.
  --version          Show version.
  --model=<model>    Selects which model to use when classifying.
  --input=<infile>   Selects which file to classify.
  --output=<outfile> Selects where to store the output.

"""
using DocOpt
args = docopt(doc, version=v"2.0.0")

using BSON

include("train.jl")

function main(model, input, output)
    @show (model, input, output)
    base = BSON.load("models/$model/base.weights")[:cpubase]
    classify = BSON.load("models/$model/classify.weights")[:cpuclass]
    labels = apply_labels_to_file(input, Chain(base, classify))
    open(output, "w") do io
        for i=1:size(labels, 1)
            println(io, join(labels[i, :], ", "))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(args["--model"], args["--input"], args["--output"])
end
