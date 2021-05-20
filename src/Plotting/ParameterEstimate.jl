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
    left_margin --> 10Plots.mm
    right_margin --> 25Plots.mm

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
            ymax = round(max(I.timeseries...),  sigdigits=3)
            ymin = round(min(I.timeseries...),  sigdigits=3)
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
            ymax = round(ymax,  sigdigits=3)
            ymin = round(ymin,  sigdigits=3)
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
        clim = max(abs.(extrema(clusterF))...)
        clims := (-clim, clim)
        println("The clustered features are:\n")
        display.(Catch22.featureDims(clusterF))
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
    pmax = round(max(I.parameters...), sigdigits=3)
    pmin = round(min(I.parameters...), sigdigits=3)
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
        ρₚ = round(corspearman(I.parameters[Int.(windowCentres)], I.estimates); sigdigits=3)
        annotations:= [(xs, ymin, text("$pmin", :black, :left, 8)),
                        (xs, ymax, text("$pmax", :black, :left, 8)),
                        (xs, (ymin+ymax)/2, text("ρ = $ρₚ", :black, :left, 11))]
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
            ρmin = round(min(ρ...), sigdigits=3)
            ρmax = round(max(ρ...), sigdigits=3)
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


# @userplot BaselineComparison
# @recipe function f(P::BaselineComparison; tswindows=(length(P.args[1].timeseries) > 10000), kwargs...)

#     𝑏 = []

#     cluster=false
#     normalisef=false
#     F = P.args[1]
#     Fₕ = P.args[2]
#     Fₗ = P.args[3]
#     legend --> false
#     #link := :x
#     grid --> false
#     layout --> @layout [
#         x
#         [a b c d]
#         [d e f g]
#     ]
#     size --> (800, 500)
#     # left_margin --> 10Plots.mm
#     # right_margin --> 25Plots.mm

#     windowCentres = round.((I.windowEdges[1:end-1] + I.windowEdges[2:end])./2)
# # ------------------------------------------ Time series ----------------------------------------- #
#     t = dims(I.timeseries, Ti).val
#     if tswindows
#         @series begin
#             seriestype := :tswindows
#             subplot := 1
#             yguide --> "Time Series"
#             xguide := ""
#             framestyle := :box
#             N --> Int(round(mean((collect(I.windowEdges[2:end]) .- collect(I.windowEdges[1:end-1])))))
#             yaxis --> nothing
#             xaxis --> nothing
#             top_margin --> 3Plots.mm
#             ymax = round(max(I.timeseries...),  sigdigits=3)
#             ymin = round(min(I.timeseries...),  sigdigits=3)
#             annotations := [(max(t...)*1.01, ymin, text("$ymin", :black, :left, 8)),
#             (max(t...)*1.01, ymax, text("$ymax", :black, :left, 8))]
#             xlims := extrema(t)
#             (t, I.timeseries)
#         end
#     else
#         @series begin
#             seriestype := :path
#             subplot := 1
#             yguide --> "Time Series"
#             xguide := ""
#             framestyle := :box
#             yaxis --> nothing
#             xaxis --> nothing
#             top_margin --> 3Plots.mm
#             seriescolor := :black
#             ymax = max(I.timeseries...)
#             ymin = min(I.timeseries...)
#             ymax -= 0.05*(ymax-ymin)
#             ymin += 0.05*(ymax-ymin)
#             ymax = round(ymax,  sigdigits=3)
#             ymin = round(ymin,  sigdigits=3)
#             annotations:= [(max(t...)*1.01, ymin, text("$ymin", :black, :left, 8)),
#             (max(t...)*1.01, ymax, text("$ymax", :black, :left, 8))]
#             xlims := extrema(t)
#             x := t
#             y := I.timeseries
#         end
#     end

# # ------------------------------------------- Features ------------------------------------------- #
#     # Need to edit the plot attributes of this subplot in the last series of this recipe
#     i = 1
#     for 𝑏 ∈ 1:ℬ
#         i += 1
#         I = infer(S, 1; normalisation=noconstantrows∘nonanrows, baseline=𝑏, kwargs...)
#         F = I.F̂
#         if cluster
#             #D = StatsBase.corspearman(F')
#             #idxs = clusterDistances(D, linkageMetric=:average, branchOrder=:optimal)
#             idxs = clusterPairwise(F, CorrDist(), linkageMetric=:average, branchOrder=:optimal, dim=1)
#             clusterF = F[idxs, :]
#         else
#             ρ = (mapslices(x -> corspearman(x, I.parameters[Int.(windowCentres)]), F, dims=2))
#             idxs = sortperm(ρ[:], rev=true)
#             clusterF = F[idxs, :]
#         end
#         if normalisef
#             clusterF = sigmoidNormalise(clusterF)
#         else
#             clusterF = normalise(clusterF, centre, 2)
#         end
#         @series begin
#             seriestype := :heatmap
#             framestyle := :box
#             colorbar --> nothing
#             ymax = length(ρ)+0.5
#             ymin = 0.5
#             ymax -= 0.05*(ymax-ymin)
#             ymin += 0.05*(ymax-ymin)
#             if cluster == false
#                 ρmin = round(min(ρ...), sigdigits=3)
#                 ρmax = round(max(ρ...), sigdigits=3)
#                 annotations:= [(max(dims(I.timeseries, Ti).val...)*1.01, ymax, text("ρ = $ρmin", :black, :left, 8)),
#                             (max(dims(I.timeseries, Ti).val...)*1.01,  ymin, text("ρ = $ρmax", :black, :left, 8))]
#             end
#             yaxis := nothing
#             xaxis := nothing
#             yflip := true
#             yguide --> "Features"
#             framestyle := :box
#             #xlims := extrema(dims(I.timeseries, Ti).val)
#             ylims := (0.5, length(idxs)+0.5)
#             subplot := i
#             seriescolor := cgrad(:RdYlBu_11, 7, categorical = true)
#             clim = max(abs.(extrema(clusterF))...)
#             clims := (-clim, clim)
#             println("The clustered features are:\n")
#             display.(Catch22.featureDims(clusterF))
#             #clusterReorder(F, CorrDist(), linkageMetric=:average, branchOrder=:optimal, dim=1)
#             (x, y, X) = (dims(I.timeseries, Ti).val[Int.(windowCentres)], 1:size(clusterF, 1), clusterF)
#         end

#         @series begin
#             seriestype := :path
#             subplot := i*2-1
#             yaxis := nothing
#             seriescolor := :black
#             xlims := extrema(t)
#             (x, y) = (t, I.parameters)
#         end
#         pmax = round(max(I.parameters...), sigdigits=3)
#         pmin = round(min(I.parameters...), sigdigits=3)
#         inset_subplots := (3, bbox(0,0,1,1))
#         @series begin
#             seriestype := :scatter
#             seriescolor := :black
#             xguide --> "Time"
#             yguide --> "Parameters"
#             framestyle := :box
#             yaxis := nothing
#             xaxis := nothing
#             background_color_inside := nothing
#             background_color_subplot := nothing
#             p = I.estimates
#             if StatsBase.corspearman(p, I.parameters[Int.(windowCentres)]) < 0
#                 @warn "It looks like the estimated parameters are the negative of the true parameters. This is not unexpected, so flipping for visualisation."
#                 p = -p
#             end
#             ymax = max(p...)
#             ymin = min(p...)
#             ymax += 0.05*(ymax-ymin)
#             ymin -= 0.05*(ymax-ymin)
#             ylims := (ymin, ymax)
#             ymax -= 0.1*(ymax-ymin)
#             ymin += 0.1*(ymax-ymin)
#             xs = length(p)*1.01
#             ρₚ = round(corspearman(I.parameters[Int.(windowCentres)], I.estimates); sigdigits=3)
#             annotations:= [(xs, ymin, text("$pmin", :black, :left, 8)),
#                             (xs, ymax, text("$pmax", :black, :left, 8)),
#                             (xs, (ymin+ymax)/2, text("ρ = $ρₚ", :black, :left, 11))]
#             xlims := (0, length(p)) # These go in centres of windows

#             (x, y) = (0.5:1:length(p)-0.5, p)
#         end
#     end
# end





