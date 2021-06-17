using HypothesisTests
# ------------------------------------------------------------------------------------------------ #
#                        Plot inference (consider changing to series recipe)                       #
# ------------------------------------------------------------------------------------------------ #
# This one takes a time series and moves all the way through to esimation, plotting relevant stuff
# Plots best in pyplot()
@userplot ParameterEstimate
@recipe function f(P::ParameterEstimate; cluster=false, tswindows=(length(P.args[1].timeseries) > 10000), normalisef=false, textcolor=:black, featurecolor=cgrad(:RdYlBu_11, 7, categorical = true))
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
    t = NonstationaryProcesses.timeDims(I.timeseries)
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
            annotations := [(max(t...)*1.01, ymin, text("$ymin", textcolor, :left, 8)),
            (max(t...)*1.01, ymax, text("$ymax", textcolor, :left, 8))]
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
            annotations:= [(max(t...)*1.01, ymin, text("$ymin",  textcolor, :left, 8)),
            (max(t...)*1.01, ymax, text("$ymax", textcolor, :left, 8))]
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
        seriescolor := featurecolor
        clim = max(abs.(extrema(clusterF))...)
        clims := (-clim, clim)
        println("The clustered features are:\n")
        display.(Catch22.featureDims(clusterF))
        #clusterReorder(F, CorrDist(), linkageMetric=:average, branchOrder=:optimal, dim=1)
        (x, y, X) = (NonstationaryProcesses.timeDims(I.timeseries)[Int.(windowCentres)], 1:size(clusterF, 1), clusterF)
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
        annotations:= [(xs, ymin, text("$pmin", textcolor, :left, 8)),
                        (xs, ymax, text("$pmax", textcolor, :left, 8)),
                        (xs, (ymin+ymax)/2, text("ρ = $ρₚ", textcolor, :left, 11))]
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
            annotations:= [(max(NonstationaryProcesses.timeDims(I.timeseries)...)*1.01, ymax, text("ρ = $ρmin", textcolor, :left, 8)),
                        (max(NonstationaryProcesses.timeDims(I.timeseries)...)*1.01,  ymin, text("ρ = $ρmax", textcolor, :left, 8))]
        end
        yaxis := nothing
        xaxis := nothing
        yflip := true
        yguide --> "Features"
        framestyle := :box
        #xlims := extrema(dims(I.timeseries, Ti).val)
        ylims := (0.5, length(idxs)+0.5)
        x = NonstationaryProcesses.timeDims(I.timeseries)[I.windowEdges]
    end

end


# ------------------------------------------------------------------------------------------------ #
#                                        Baseline Comparison                                       #
# ------------------------------------------------------------------------------------------------ #

@userplot BaselineComparison
@recipe function f(P::BaselineComparison; features=catch24, parameters=1, var=1, tswindows=(length(timeseries(P.args[1])) > 10000), orthonormalise=false)
    S = P.args[1]
    Fₕ = P.args[2]
    Fₗ = P.args[3]

    baselines = [
        "Standardised"
        "Low dim. baseline"
        "High dim. baseline"
        "Interval baseline"
    ]

    if orthonormalise == :orthonormalise || (orthonormalise isa Bool && orthonormalise)
        # Orthogonalise and then scale
        𝑜 = orthogonaliseto(Fₕ, principalcomponents)
        I_a = [
            infer(S, var; parameters, features, baseline=standardbaseline(), normalisation=𝑜), # No baseline
            infer(S, var; parameters, features, baseline=lowbaseline(𝑜(Fₗ)), normalisation=𝑜), # Low
            infer(S, var; parameters, features, baseline=highbaseline(𝑜(Fₕ)), normalisation=𝑜), # High
            infer(S, var; parameters, features, baseline=intervalbaseline(𝑜(Fₗ), 𝑜(Fₕ)), normalisation=𝑜)  # Both
        ]
    elseif orthonormalise == :orthogonalise
        I_a = [
            infer(S, var; parameters, features, baseline=orthogonaliseto(standardbaseline()(Fₕ)), normalisation=standardbaseline()), # No baseline
            infer(S, var; parameters, features, baseline=orthogonaliseto(lowbaseline(Fₗ)(Fₕ)), normalisation=lowbaseline(Fₗ)), # Low
            infer(S, var; parameters, features, baseline=orthogonaliseto(highbaseline(Fₕ)(Fₕ)), normalisation=highbaseline(Fₕ)), # High
            infer(S, var; parameters, features, baseline=orthogonaliseto(intervalbaseline(Fₗ, Fₕ)(Fₕ)), normalisation=intervalbaseline(Fₗ, Fₕ))  # Both
        ]
    elseif orthonormalise == :ranksum_orthogonalise
        # ! Not done ------------------------------->
        𝑜 = orthogonaliseto(Fₕ, principalcomponents)
        F̂ₕ, F̂ₗ = 𝑜.((Fₕ, Fₗ))
        p = zeros(size(F̂ₕ, 1))
        for i = 1:size(F̂ₕ, 1)
            if all(F̂ₗ[i, :] .== F̂ₗ[i, 1]) # The low baseline is constant, so use a one-sample t-test
                p[i] = pvalue(OneSampleTTest(F̂ₗ[i, :], F̂ₕ[i, :]))
            else
                p[i] = pvalue(MannWhitneyUTest(F̂ₗ[i, :], F̂ₕ[i, :]))
            end
        end
        display(histogram(Fₕ))
        show(p)

        I_a = [
            infer(S, var; parameters, features, baseline=orthogonaliseto(standardbaseline()(Fₕ)), normalisation=standardbaseline()), # No baseline
            infer(S, var; parameters, features, baseline=orthogonaliseto(lowbaseline(Fₗ)(Fₕ)), normalisation=lowbaseline(Fₗ)), # Low
            infer(S, var; parameters, features, baseline=orthogonaliseto(highbaseline(Fₕ)(Fₕ)), normalisation=highbaseline(Fₕ)), # High
            infer(S, var; parameters, features, baseline=orthogonaliseto(intervalbaseline(Fₗ, Fₕ)(Fₕ)), normalisation=intervalbaseline(Fₗ, Fₕ))  # Both
        ]
    elseif orthonormalise == :truncate3
        # ! Not done ------------------------------->
        𝑜 = orthogonaliseto(Fₕ, principalcomponents)
        F̂ₕ, F̂ₗ = 𝑜.((Fₕ, Fₗ))

        idxs = partialsortperm(vec(StatsBase.std(F̂ₕ, dims=2)), 1:3, rev=true)
        
        F̂ₕ = F̂ₕ[idxs, :]
        F̂ₗ = F̂ₗ[idxs, :]

        I_a = [
            infer(S, var; parameters, features, baseline=standardbaseline()∘(X -> X[idxs, :]), normalisation=orthogonaliseto(Fₕ), filter=_self), # No baseline
            infer(S, var; parameters, features, baseline=lowbaseline(F̂ₗ)∘(X -> X[idxs, :]), normalisation=orthogonaliseto(Fₕ), filter=_self), # Low
            infer(S, var; parameters, features, baseline=highbaseline(F̂ₕ)∘(X -> X[idxs, :]), normalisation=orthogonaliseto(Fₕ), filter=_self), # High
            infer(S, var; parameters, features, baseline=intervalbaseline(F̂ₗ, F̂ₕ)∘(X -> X[idxs, :]), normalisation=orthogonaliseto(Fₕ), filter=_self)  # Both
        ]
    elseif orthonormalise == :totalcovariance
        function projectedtotalcovariance(𝑏::Function)
            M = principalcomponents(Array(𝑏(Fₕ)))
            𝑜 = orthogonaliseto(𝑏(Fₕ), principalcomponents)
            Σₕ² = StatsBase.cov(𝑏(Fₕ), dims=2)
            T = Array(sum(abs.(Σₕ²), dims=2))
            T′ = projection(M)'*Diagonal(T[:])*projection(M) # Don't want to subtract means, just want the rotation
            # Will want to rethink this formula, since we are now ignoring the off-diagonal terms.
            #T′ = sqrt.(diag(T′))
            g = X -> FeatureMatrix(inv(T′)*Array(X), getnames(X))
            return g∘𝑜
        end
        I_a = [
            infer(S, var; parameters, features, baseline=projectedtotalcovariance(standardbaseline()), normalisation=standardbaseline(), filter=_self), # No baseline
            infer(S, var; parameters, features, baseline=projectedtotalcovariance(lowbaseline(Fₗ)), normalisation=lowbaseline(Fₗ), filter=_self), # Low
            infer(S, var; parameters, features, baseline=projectedtotalcovariance(highbaseline(Fₕ)), normalisation=highbaseline(Fₕ), filter=_self), # High
            infer(S, var; parameters, features, baseline=projectedtotalcovariance(intervalbaseline(Fₗ, Fₕ)), normalisation=intervalbaseline(Fₗ, Fₕ), filter=_self)  # Both
        ]
    elseif orthonormalise == :dependencyscaling
        #! This should be identical to :totalcovariance
        function dependencyscaling(𝑏, Fₕ)
            Fₕ′ = 𝑏(Fₕ)
            Σₕ² = StatsBase.cov(Array(Fₕ′), dims=2)
            𝑜 = orthogonaliseto(Fₕ′, principalcomponents)
            𝐧 = sum(abs.(Array(Σₕ²)), dims=2)
            N⁻¹ = FeatureMatrix(inv(sqrt(Diagonal(𝐧[:]))), getnames(Fₕ′))
            return F -> FeatureMatrix(𝑜(N⁻¹*𝑏(F)), getnames(F))
        end
        I_a = [
            infer(S, var; parameters, features, baseline=dependencyscaling(standardbaseline(), Fₕ)), # No baseline
            infer(S, var; parameters, features, baseline=dependencyscaling(lowbaseline(Fₗ), Fₕ)), # Low
            infer(S, var; parameters, features, baseline=dependencyscaling(highbaseline(Fₕ), Fₕ)), # High
            infer(S, var; parameters, features, baseline=dependencyscaling(intervalbaseline(Fₗ, Fₕ), Fₕ))  # Both
        ]
    elseif orthonormalise == :dependencyscalingnorotation
        #! This should be identical to :dependencyscaling
        function dependencyscalingnorotation(𝑏, Fₕ)
            Fₕ′ = 𝑏(Fₕ)
            Σₕ² = StatsBase.cov(Array(Fₕ′), dims=2)
            𝑜 = orthogonaliseto(Fₕ′, principalcomponents)
            𝐧 = sum(abs.(Array(Σₕ²)), dims=2)
            N⁻¹ = FeatureMatrix(inv(sqrt(Diagonal(𝐧[:]))), getnames(Fₕ′))
            return F -> FeatureMatrix(N⁻¹*𝑏(F), getnames(F))
        end
        I_a = [
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(standardbaseline(), Fₕ)), # No baseline
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(lowbaseline(Fₗ), Fₕ)), # Low
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(highbaseline(Fₕ), Fₕ)), # High
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(intervalbaseline(Fₗ, Fₕ), Fₕ))  # Both
        ]
    elseif orthonormalise == :whiten
        function whitento(𝑏, Fₕ)
            Fₕ′ = 𝑏(Fₕ)
            𝑜 = orthogonaliseto(Fₕ′, principalcomponents)
            𝐧 = sum(abs.(Array(Σₕ²)), dims=2)
            N⁻¹ = FeatureMatrix(inv(sqrt(Diagonal(𝐧[:]))), getnames(Fₕ′))
            return F -> FeatureMatrix(N⁻¹*𝑏(F), getnames(F))
        end
        I_a = [
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(standardbaseline(), Fₕ)), # No baseline
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(lowbaseline(Fₗ), Fₕ)), # Low
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(highbaseline(Fₕ), Fₕ)), # High
            infer(S, var; parameters, features, baseline=dependencyscalingnorotation(intervalbaseline(Fₗ, Fₕ), Fₕ))  # Both
        ]
    elseif orthonormalise == :errororthonormalise
            # Orthogonalise and then scale, but using the error interval for the high and zero dim scaling
            𝑜 = orthogonaliseto(Fₕ, principalcomponents)
            I_a = [
                infer(S, var; parameters, features, baseline=standardbaseline(), normalisation=𝑜), # No baseline
                infer(S, var; parameters, features, baseline=lowbaseline(𝑜(Fₗ)), normalisation=𝑜), # Low
                infer(S, var; parameters, features, baseline=highbaseline(𝑜(Fₕ)), normalisation=𝑜), # High
                infer(S, var; parameters, features, baseline=errorintervalbaseline(𝑜(Fₗ), 𝑜(Fₕ)), normalisation=𝑜)  # Both
            ]
    else
        I_a = [
            infer(S, var; parameters, features, baseline=standardbaseline()), # No baseline
            infer(S, var; parameters, features, baseline=lowbaseline(Fₗ)), # Low
            infer(S, var; parameters, features, baseline=highbaseline(Fₕ)), # High
            infer(S, var; parameters, features, baseline=intervalbaseline(Fₗ, Fₕ))  # Both
        ]
    end
    cluster=false
    normalisef=false
    legend --> false
    #link := :x
    grid --> false
    layout --> @layout [
        x{0.2h}
        [a; b] [c; d]
        [d; e] [f; g]
    ]
    size --> (1000, 1000)
    left_margin --> 20Plots.mm
    right_margin --> 20Plots.mm

    # Do a quick inference for quick access to time series
    I = infer(S, var; parameters, features, baseline=_self, normalisation=_self)

    windowCentres = round.((I.windowEdges[1:end-1] + I.windowEdges[2:end])./2)
# ------------------------------------------ Time series ----------------------------------------- #
    t = NonstationaryProcesses.timeDims(I.timeseries)
    if tswindows
        @series begin
            seriestype := :tswindows
            subplot := 1
            # yguide --> "Time Series"
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
            # yguide --> "Time Series"
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
    bb = [1, 5, 3, 7]
    bbb = [2, 6, 4, 8]
    bbbb = [9, 10, 11, 12]
    inset_subplots := [(x+1, bbox(0,0,1,1)) for x ∈ bbb]
    for b ∈ 1:4
        I = I_a[b]
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
            framestyle := :box
            title := baselines[b]
            colorbar --> nothing
            ymax = length(ρ)+0.5
            ymin = 0.5
            ymax -= 0.05*(ymax-ymin)
            ymin += 0.05*(ymax-ymin)
            if !cluster
                ρmin = round(min(ρ...), sigdigits=3)
                ρmax = round(max(ρ...), sigdigits=3)
                annotations:= [(max(NonstationaryProcesses.timeDims(I.timeseries)...)*1.01, ymax, text("ρ = $ρmin", :black, :left, 8)),
                            (max(NonstationaryProcesses.timeDims(I.timeseries)...)*1.01,  ymin, text("ρ = $ρmax", :black, :left, 8))]
            end
            yaxis := nothing
            xaxis := nothing
            yflip := true
            # if b == 1
            #     yguide := "Features"
            # end
            framestyle := :box
            #xlims := extrema(dims(I.timeseries, Ti).val)
            ylims := (0.5, length(idxs)+0.5)
            subplot := bb[b]+1
            seriescolor := cgrad(:RdYlBu_11, 7, categorical = true)
            clim = max(abs.(extrema(clusterF))...)
            clims := (-clim, clim)
            println("The clustered features are:")
            display.(Catch22.featureDims(clusterF))
            print("\n")
            #clusterReorder(F, CorrDist(), linkageMetric=:average, branchOrder=:optimal, dim=1)
            (x, y, X) = (NonstationaryProcesses.timeDims(I.timeseries)[Int.(windowCentres)], 1:size(clusterF, 1), clusterF)
        end

        @series begin
            seriestype := :path
            subplot := bbb[b]+1
            yaxis := nothing
            seriescolor := :black
            xlims := extrema(t)
            (x, y) = (t, I.parameters)
        end
        pmax = round(max(I.parameters...), sigdigits=3)
        pmin = round(min(I.parameters...), sigdigits=3)
        @series begin
            seriestype := :scatter
            seriescolor := :black
            # xguide --> "Time"
            subplot := bbbb[b]+1
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
            markercolor := cgrad([:crimson, :forestgreen], LinRange(0, 1, 256))[round(Int, abs(ρₚ)*255)+1]
            markerstrokewidth := 0.0
            annotations:= [(xs, ymin, text("$pmin", :black, :left, 8)),
            (xs, ymax, text("$pmax", :black, :left, 8)),
            (xs, (ymin+ymax)/2, text("ρ = $ρₚ", :black, :left, 8))]
            # if bbb[b] < 5
            #     yguide := "Parameters"
            # end
            xlims := (0, length(p)) # These go in centres of windows

            (x, y) = (0.5:1:length(p)-0.5, p)
        end
    end
end





