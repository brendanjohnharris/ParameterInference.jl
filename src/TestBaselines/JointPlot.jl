
@userplot Joint
@recipe function f(g::Joint; colormode=:columns)
    @assert length(g.args) == 2 && g.args[2] isa AbstractMatrix
    x, y = g.args
    for (n, r) ∈ y |> eachrow |> enumerate
        @series begin
            seriestype := :violin
            if colormode == :rows
                (seriescolor := n)
            else
                (seriescolor := RGBA(0, 0, 0, 0.1))
            end
            linecolor := RGBA(0, 0, 0, 0.2)
            label := nothing
            x := n
            y := r
        end
    end
    for (m, s) ∈ y |> eachcol |> enumerate
        @series begin
            seriestype := :path
            linealpha --> 0.5
            if colormode == :columns
                (seriescolor := m)
            else
                (seriescolor := :gray)
            end
            label := nothing
            x := x
            y := s
        end
        @series begin
            seriestype := :scatter
            markeralpha --> 0.5
            if colormode == :columns
                (seriescolor := m)
            else
                (seriescolor := :gray)
            end
            label := nothing
            xticks := (1:length(x), x)
            x := x
            y := s
        end
    end
end
