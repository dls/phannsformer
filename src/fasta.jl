using FastaIO
import Serialization

"Similar to tokenization in NLP. Defines our mapping from amino acids to embeddable integers"
mapping = Dict(
            'M' => 1,
            'S' => 2,
            'I' => 3,
            'V' => 4,
            'A' => 5,
            'L' => 6,
            'R' => 7,
            'T' => 8,
            'P' => 9,
            'E' => 10,
            'Y' => 11,
            'K' => 12,
            'H' => 13,
            'D' => 14,
            'Q' => 15,
            'W' => 16,
            'F' => 17,
            'G' => 18,
            'N' => 19,
            'C' => 20,
            'X' => 21,
            'J' => 22,
            'B' => 23,
            'Z' => 24)

"Load ORFs from a file, each converted to a UInt8 vector"
function byte_mapped_sequences_in_file(file)
    seqs = []

    i = 0
    FastaReader(file) do fr
        for (desc, seq) in fr
            mapped = UInt8[]
            for r=seq
                if haskey(mapping, r)
                    push!(mapped, mapping[r])
                end
            end
            push!(seqs, mapped)
        end
    end
    return seqs
end