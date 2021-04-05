using Plots
using DimensionalData
import StatsBase


# ------------------------------------------------------------------------------------------------ #
#                                       Plot a feature matrix                                      #
# ------------------------------------------------------------------------------------------------ #
# @recipe function f(::Type{Val{:featurematrix}}, x, y, z)
#     # seriescolor := cgrad(:RdYlBu_11, 7, categorical = true)
#     #yticks := nothing
#     #colorbar --> nothing
#     # xticks := (x, string.(x))
#     # yticks := (y, string.(y))
#     seriestype := :heatmap
#     x := x
#     y := y
#     z := z
#     #(x, y, z)
#     ()
# end

# @recipe function f(fm::Type{Val{:featurematrix}}, F::DimArray, args...)
#     seriestype := featurematrix
#     features := String.(dims(F, :feature).val)
#     timeseries := String.(dims(F, :timeseries).val)
#     F --> Array(F)
#     ()
# end




# ------------------------------------------------------------------------------------------------ #
#                        Plot inference (consider changing to series recipe)                       #
# ------------------------------------------------------------------------------------------------ #
# This one takes a time series and moves all the way through to esimation, plotting relevant stuff
# Plots best in pyplot()
@userplot ParameterEstimate
@recipe function f(P::ParameterEstimate; cluster=false, tswindows=(length(P.args[1].timeseries) > 10000), normalisef=true)
    I = P.args[1]
    legend --> false
    #link := :x
    grid --> false
    layout --> @layout [
        x
        F
        P
    ]
    size --> (800, 500)
    left_margin --> 5Plots.mm
    right_margin --> 15Plots.mm

    windowCentres = round.((I.windowEdges[1:end-1] + I.windowEdges[2:end])./2)
# ------------------------------------------ Time series ----------------------------------------- #
    t = dims(I.timeseries, Ti).val
    if tswindows
        @series begin
            seriestype := :tswindows
            subplot := 1
            yguide --> "Time Series"
            xguide := ""
            framestyle := :box
            N --> Int(round(mean((collect(I.windowEdges[2:end]) .- collect(I.windowEdges[1:end-1])))))
            yaxis --> nothing
            xaxis --> nothing
            top_margin --> 3Plots.mm
            ymax = round(max(I.timeseries...),  sigdigits=2)
            ymin = round(min(I.timeseries...),  sigdigits=2)
            annotations := [(max(t...)*1.01, ymin, text("$ymin", :black, :left, 8)),
            (max(t...)*1.01, ymax, text("$ymax", :black, :left, 8))]
            xlims := extrema(t)
            (t, I.timeseries)
        end
    else
        @series begin
            seriestype := :path
            subplot := 1
            yguide --> "Time Series"
            xguide := ""
            framestyle := :box
            yaxis --> nothing
            xaxis --> nothing
            top_margin --> 3Plots.mm
            seriescolor := :black
            ymax = max(I.timeseries...)
            ymin = min(I.timeseries...)
            ymax -= 0.05*(ymax-ymin)
            ymin += 0.05*(ymax-ymin)
            ymax = round(ymax,  sigdigits=2)
            ymin = round(ymin,  sigdigits=2)
            annotations:= [(max(t...)*1.01, ymin, text("$ymin", :black, :left, 8)),
            (max(t...)*1.01, ymax, text("$ymax", :black, :left, 8))]
            xlims := extrema(t)
            x := t
            y := I.timeseries
        end
    end

# ------------------------------------------- Features ------------------------------------------- #
    # Need to edit the plot attributes of this subplot in the last series of this recipe
    F = I.F̂
    if cluster
        #D = StatsBase.corspearman(F')
        #idxs = clusterDistances(D, linkageMetric=:average, branchOrder=:optimal)
        idxs = clusterPairwise(F, CorrDist(), linkageMetric=:average, branchOrder=:optimal, dim=1)
        clusterF = F[idxs, :]
    else
        ρ = (mapslices(x -> corspearman(x, I.parameters[Int.(windowCentres)]), F, dims=2))
        idxs = sortperm(ρ[:], rev=true)
        clusterF = F[idxs, :]
    end
    if normalisef
        clusterF = sigmoidNormalise(clusterF)
    else
        clusterF = normalise(clusterF, centre, 2)
    end
    @series begin
        seriestype := :heatmap
        subplot := 2
        seriescolor := cgrad(:RdYlBu_11, 7, categorical = true)
        println("The clustered features are:\n")
        display.(dims(F, :feature).val[idxs[end:-1:1]])
        #clusterReorder(F, CorrDist(), linkageMetric=:average, branchOrder=:optimal, dim=1)
        (x, y, X) = (dims(I.timeseries, Ti).val[Int.(windowCentres)], 1:size(clusterF, 1), clusterF)
    end



# ------------------------------------------ Parameters ------------------------------------------ #

    @series begin
        seriestype := :path
        subplot := 3
        yaxis := nothing
        seriescolor := :black
        xlims := extrema(t)
        (x, y) = (t, I.parameters)
    end
    pmax = round(max(I.parameters...), sigdigits=2)
    pmin = round(min(I.parameters...), sigdigits=2)
    inset_subplots := (3, bbox(0,0,1,1))
    @series begin
        seriestype := :scatter
        seriescolor := :black
        xguide --> "Time"
        yguide --> "Parameters"
        framestyle := :box
        yaxis := nothing
        xaxis := nothing
        background_color_inside := nothing
        background_color_subplot := nothing
        p = I.estimates
        if StatsBase.corspearman(p, I.parameters[Int.(windowCentres)]) < 0
            @warn "It looks like the estimated parameters are the negative of the true parameters. This is not unexpected, so flipping for visualisation."
            p = -p
        end
        ymax = max(p...)
        ymin = min(p...)
        ymax += 0.05*(ymax-ymin)
        ymin -= 0.05*(ymax-ymin)
        ylims := (ymin, ymax)
        ymax -= 0.1*(ymax-ymin)
        ymin += 0.1*(ymax-ymin)
        xs = length(p)*1.01
        annotations:= [(xs, ymin, text("$pmin", :black, :left, 8)),
        (xs, ymax, text("$pmax", :black, :left, 8))]
        xlims := (0, length(p)) # These go in centres of windows

        (x, y) = (0.5:1:length(p)-0.5, p)
    end

# -------------------------------- Add windows to the feature plot ------------------------------- #
    @series begin
        seriescolor := :black
        seriestype := :vline
        subplot := 2
        framestyle := :box
        colorbar --> nothing
        ymax = length(ρ)+0.5
        ymin = 0.5
        ymax -= 0.05*(ymax-ymin)
        ymin += 0.05*(ymax-ymin)
        if cluster == false
            ρmin = round(min(ρ...), sigdigits=2)
            ρmax = round(max(ρ...), sigdigits=2)
            annotations:= [(max(dims(I.timeseries, Ti).val...)*1.01, ymax, text("ρ = $ρmin", :black, :left, 8)),
                        (max(dims(I.timeseries, Ti).val...)*1.01,  ymin, text("ρ = $ρmax", :black, :left, 8))]
        end
        yaxis := nothing
        xaxis := nothing
        yflip := true
        yguide --> "Features"
        framestyle := :box
        #xlims := extrema(dims(I.timeseries, Ti).val)
        ylims := (0.5, length(idxs)+0.5)
        x = dims(I.timeseries, Ti).val[I.windowEdges]
    end

end




@userplot DimensionalityEstimate
@recipe function f(D::DimensionalityEstimate)
    I = D.args[1]
    legend --> false
    #link := :x
    right_margin --> 10Plots.mm
    grid --> false
    size --> (1000, 400)
    framestyle := :box
    layout --> @layout [ev rv]

    σ² = residualVariance(I.model, I.F̂)
    ξ² = explainedVariance(I.model)
    @series begin
        seriestype := :path
        markersize --> 5
        subplot := 1
        marker --> :circle
        label --> nothing
        seriescolor --> :black
        xguide --> "Principal Components"
        yguide --> "Residual Variance"
        (x, y) = (0:length(σ²), [1, σ²...])
    end

    @series begin
        seriestype := :path
        markersize --> 5
        subplot := 2
        marker --> :circle
        label --> nothing
        seriescolor --> :red
        xguide --> "Principal Components"
        yguide --> "Explained Variance"
        (x, y) = (0:length(ξ²), [0, ξ²...])
    end
end





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
        # anum = (round((x[idxs[s][end]] + x[idxs[s][1]])/2, sigdigits=2))
        # if dispxs[end] > 99
        #     anum = Int(anum)
        # end
        # annotations := [(annx, min(y...)-2*scale, text("$anum", :black, :centre, 8))]
        if s < length(idxs)
            barx = (max(dispxs[nn:nn+N-1]...) + min(dispxs[(nn+N):(nn+2*N-1)]...))/2
            @series begin
                seriestype := :path
                linewidth := 3
                seriescolor := :red
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