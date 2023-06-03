using Pkg
import Serialization

if haskey(Pkg.installed(), "PyCall")
    using PyCall

    sio = pyimport("Bio.SeqIO")

    mapping = Dict("M" => 1,
                   "S" => 2,
                   "I" => 3,
                   "V" => 4,
                   "A" => 5,
                   "L" => 6,
                   "R" => 7,
                   "T" => 8,
                   "P" => 9,
                   "E" => 10,
                   "Y" => 11,
                   "K" => 12,
                   "H" => 13,
                   "D" => 14,
                   "Q" => 15,
                   "W" => 16,
                   "F" => 17,
                   "G" => 18,
                   "N" => 19,
                   "C" => 20,
                   "X" => 21,
                   "J" => 22,
                   "B" => 23,
                   "Z" => 24)

    function sequences_in_file(file)
        seqs = [UInt8[]]

        println(file)
        i = 0
        for record in sio.parse(file, "fasta")
            mapped = [mapping[r] for r=record.seq]
            println((i += 1))
            push!(seqs, mapped)
        end

        return seqs[2:end]
    end

    function parseem()
        seqs = [UInt8[]]
        cats = UInt8[]

        categories = ["major_capsid", "minor_capsid", "baseplate", "major_tail",
                      "minor_tail", "portal", "tail_fiber", "shaft", "collar", "HTJ", "other"]

        # categories = ["HTJ", "baseplate", "collar", "major_capsid", "major_tail",
        #               "minor_capsid", "minor_tail", "other", "portal", "shaft", "tail_fiber"]

        for c in 1:length(categories)
            for n in 1:11
                println("$(n)_$(categories[c]).fasta")
                for record in sio.parse("$(n)_$(categories[c]).fasta", "fasta")
                    mapped = [mapping[r] for r=record.seq]
                    push!(seqs, mapped)
                    push!(cats, c)
                end
            end
        end

        return PhANNsDataset(cats, map(length, seqs[2:end]), seqs[2:end])
    end
end

function load_cache_or_regen()
    if isfile("phanns_cache.data")
        return Serialization.deserialize("phanns_cache.data")
    else
        phanns_data = parseem()
        Serialization.serialize("phanns_cache.data", phanns_data)
        return phanns_data
    end
end

phanns_data = load_cache_or_regen()
