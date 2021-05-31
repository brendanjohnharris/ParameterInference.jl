
@recipe function f(::Type{Val{:tswindows}}, x, y, z; windows=5, N=min(length(x)÷10, 1000))
    L = length(y)
    xlims, ylims = (extrema(x), extrema(y))
    scale = (ylims[2] - ylims[1])*0.1
    ylims = (ylims[1] - scale, ylims[2] + scale)
    legend --> false
    ylims := ylims
    #seriescolor --> :black
    #grid --> false


    # We want to take a long time series, choose some windows of length N and plot these adjacently
    dL = (L÷(2*windows))
    idxs = [2*a*dL+dL-Int(floor(N/2)):1:2*a*dL+dL+Int(floor(N/2)-1) for a ∈ (1:windows).-1]
    dispxs = LinRange(0, max(x...), length(vcat(idxs...)))
    nn = -N+1
    for s in 1:length(idxs)
        nn += N
        # if nn-N < 1
        #     annx = (dispxs[nn+N])/2
        # elseif nn+N > length(dispxs)
        #     annx = (dispxs[nn] + dispxs[end])/2
        #     println(dispxs[nn])
        # else
        #     annx = (dispxs[nn+N] + dispxs[nn])/2
        # end
        # anum = (round((x[idxs[s][end]] + x[idxs[s][1]])/2, sigdigits=3))
        # if dispxs[end] > 99
        #     anum = Int(anum)
        # end
        # annotations := [(annx, min(y...)-2*scale, text("$anum", :black, :centre, 8))]
        if s < length(idxs)
            barx = (max(dispxs[nn:nn+N-1]...) + min(dispxs[(nn+N):(nn+2*N-1)]...))/2
            @series begin
                seriestype := :path
                linewidth := 3
                linecolor := :red
                x := [barx, barx]
                y := [min(y...)-scale, max(y...)+scale]
            end
        end
        @series begin
            seriestype := :path
            #framestyle --> :box
            #xlims --> xlims
            #ylims --> ylims
            seriescolor := :black
            x := dispxs[nn:nn+N-1]
            y := y[idxs[s]]
        end
    end
end

# ------------------------------------------------------------------------------------------------ #
#                              Function for plotting array of rasters                              #
# ------------------------------------------------------------------------------------------------ #

@userplot RasterArray
@recipe function f(g::RasterArray)
    @assert length(g.args) == 3 && g.args[3] isa AbstractMatrix
    seriestype := :rasterarray
    mat = g.args[3]
    g.args[1], g.args[2], Surface(mat)
end

@recipe function f(::Type{Val{:rasterarray}}, x, y, z)
    # Currently only works with gr()
    x, y, T = x, y, z.surf
    T = normalise(T, standardise, 2)
    if size(T, 1) > size(T, 2)
        @warn "Seems like a lot of time series, sure you didn't mean to transpose?"
    end
    uy = unique(y)
    ux = unique(x)
    yax = (sort∘unique)(y) # Must have equally spaced y's and x's
    xax = (sort∘unique)(x)
    P = Array{Matrix{Float64}}(undef, (length(yax), length(xax)))
    for 😄 ∈ CartesianIndices(P)
        (i, j) = Tuple(😄)
        P[😄] = T[(x.==xax[j]) .& (y.==yax[i]), :]
    end
    P .= reshape.(P, 1, size(T, 2))
    if size(P, 2) > 1
        P = [hcat(P[i, :]...) for i ∈ 1:size(P, 1)]
    end
    if size(P, 1) > 1
        P = vcat(P...)
    end
    seriestype := :heatmap
    xticks --> (LinRange(0, size(P, 2), length(ux)+1)[1:end-1].+size(P, 2)/length(ux)/2, ux)
    yticks --> (1:size(P, 1), uy)
    xlims := (0.5, size(P, 2)+0.5)
    ylims := (0.5, size(P, 1)+0.5)
    y := collect(1:size(P, 1))
    x := collect(1:size(P, 2))
    z := Surface(P)
    ()
end

# @userplot TSRasterArray
# @recipe function f(g::TSRasterArray)
#     @assert length(g.args) == 3 && typeof(g.args[3]) <: AbstractMatrix
#     seriestype := :tsrasterarray
#     mat = g.args[3]
#     g.args[1], g.args[2], Surface(mat)
# end
# @shorthands tsrasterarray
# @userplot TSRasterArray
# @recipe function f(g::TSRasterArray)
#     #x, y, z = plotattributes[:x], plotattributes[:y], plotattributes[:y].surf
#     x = g.args[1]
#     y = g.args[2]
#     z = g.args[3]
#     @series begin
#         seriestype := rasterarray
#         (x, y, z)
#     end
# end