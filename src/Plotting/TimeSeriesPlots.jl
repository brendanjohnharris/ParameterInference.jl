
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
