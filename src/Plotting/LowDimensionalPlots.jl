import StatsBase.cov
# ------------------------------------------------------------------------------------------------ #
#          Run a pca on the (time series of a) feature matrix and plot the first two PC's          #
# ------------------------------------------------------------------------------------------------ #
@userplot LowDim2
@recipe function f(h::LowDim2; method=principalcomponents, classlabels=false, features=Catch22.featureDims(h.args[1]))
    # show(size(F))
    F = h.args[1]
    # if h.args > 1
    #     Fs = h.args[2:end]
    # end
    fs = Catch22.featureDims(F)
    seriestype := scatter
    M = project(standardise(Array(F)), method)
    F′ = embed(M, standardise(Array(F)), 1:2)

    if M isa MultivariateStats.PCA
        PV = principalvars(M)
        PV = PV[1:2]./sum(PV)
        wmax = mapslices(x -> sum(abs.(x)), PCfeatureWeights(M, 1:2), dims=1)
        weights, fis = findmax(PCfeatureWeights(M)[:, 1:2], dims=1)
        PV = [round(x*100, sigdigits=2) for x ∈ PV]
        w = [round(100*weights[i]/wmax[i], sigdigits=2) for i ∈ 1:length(weights)]
        #xguide --> "$(PV[1])% Variance ($(w[1])% $(fs[fis[1]]))"
        #yguide --> "$(PV[2])% Variance ($(w[2])% $(fs[fis[1]]))"
        xguide --> "PC1 ($(PV[1])%)"
        yguide --> "PC2 ($(PV[2])%)"
    end


    if classlabels isa Bool && classlabels
        classlabels = Catch22.timeseriesDims(F)
        uniquelabels = unique(classlabels)

        for c ∈ uniquelabels
            @series begin
                idxs = classlabels .== c
                label := string(c)
                (x, y) = (F′[1, idxs], F′[2, idxs])
            end
        end
    else
        @series begin
            (x, y) = (F′[1, :], F′[2, :])
        end
        # if length(h.args) > 1
        #     for F ∈ Fs
        #         @series begin
        #             F = embed(M, standardise(Array(F), mean(Array(F), dims=2), StatsBase.std(Array(F), dims=2)), 1:2)
        #             (x, y) = (F[1, :], F[2, :])
        #         end
        #     end
        # end
    end
end



@userplot LowDim3
@recipe function f(h::LowDim3; method=principalcomponents, classlabels=true, features=Catch22.featureDims(h.args[1]))
    # show(size(F))
    F = h.args[1]
    fs = Catch22.featureDims(F)
    seriestype := scatter
    M = project(standardise(Array(F)), method)
    F′ = embed(M, standardise(Array(F)), 1:3)

    markersize --> 2
    markerstrokewidth --> 0
    size --> (600, 400)
    if M isa MultivariateStats.PCA
        PV = principalvars(M)
        PV = PV[1:3]./sum(PV)
        wmax = mapslices(x -> sum(abs.(x)), PCfeatureWeights(M)[:, 1:3], dims=1)
        weights, fis = findmax(PCfeatureWeights(M)[:, 1:3], dims=1)
        PV = [round(x*100, sigdigits=2) for x ∈ PV]
        w = [round(100*weights[i]/wmax[i], sigdigits=2) for i ∈ 1:length(weights)]
        #xguide --> "$(PV[1])% Variance ($(w[1])% $(fs[fis[1]]))"
        #yguide --> "$(PV[2])% Variance ($(w[2])% $(fs[fis[1]]))"
        xguide --> "PC1 ($(PV[1])%)"
        yguide --> "PC2 ($(PV[2])%)"
        zguide --> "PC3 ($(PV[3])%)"
    end


    if classlabels isa Bool && classlabels
        classlabels = Catch22.timeseriesDims(F)
        uniquelabels = unique(classlabels)

        for c ∈ uniquelabels
            @series begin
                idxs = classlabels .== c
                label := string(c)
                (x, y) = (F′[1, idxs], F′[2, idxs], F′[3, idxs])
            end
        end
    else
        @series begin
            (x, y) = (F′[1, :], F′[2, :], F′[3, :])
        end
    end
end


# ---------------- Plot high, test and zero-dim datasets and in the PC2 same space --------------- #
@userplot LowDimBaselines
@recipe function f(h::LowDimBaselines; method=principalcomponents, ellipsealpha=0.2, doscatter=true, grouplabels=false)
    # show(size(F))
    F = h.args[1:3] # We will whiten according to the first input

    seriestype := scatter
    M = project(F[1], method)
    F′ = embed.((M,), F, (1:2,))

    # if M isa MultivariateStats.PCA
    #     PV = principalvars(M)
    #     PV = PV[1:2]./sum(PV)
    #     wmax = mapslices(x -> sum(abs.(x)), PCfeatureWeights(M, 1:2), dims=1)
    #     weights, fis = findmax(PCfeatureWeights(M)[:, 1:2], dims=1)
    #     PV = [round(x*100, sigdigits=2) for x ∈ PV]
    #     w = [round(100*weights[i]/wmax[i], sigdigits=2) for i ∈ 1:length(weights)]
    #     #xguide --> "$(PV[1])% Variance ($(w[1])% $(fs[fis[1]]))"
    #     #yguide --> "$(PV[2])% Variance ($(w[2])% $(fs[fis[1]]))"
    #     xguide --> "PC1 ($(PV[1])%)"
    #     yguide --> "PC2 ($(PV[2])%)"
    # end

    color_palette = [:cornflowerblue, :crimson, :forestgreen]
    color_palette --> color_palette
    for i = 1:3
        @series begin
            markerstrokewidth --> 0
            label := nothing
            seriescolor --> color_palette[i]
            markeralpha --> exp(-0.002*size(F′[i], 2))
            (x, y) = (F′[i][1, :], F′[i][2, :])
        end
        @series begin
            seriesalpha := ellipsealpha
            seriescolor := color_palette[i]
            seriestype := :shape
            if grouplabels isa Array
                label := grouplabels[i]
            else
                label := grouplabels
            end
            linewidth := 2
            linecolor := color_palette[i]
            linealpha := 1.0
            framestyle --> :box
            μ, S = StatsPlots._covellipse_args(Array.((mean(F′[i], dims=2)[:], cov((F′[i]), dims=2))); n_std=1.5)
            θ = range(0, 2π; length=1000)
            A = S * [cos.(θ)'; sin.(θ)']
            (μ[1] .+ A[1,:], μ[2] .+ A[2,:])
        end
    end
end

@userplot BaselineScatters
@recipe function f(h::BaselineScatters; ellipsealpha=0.2, doscatter=true, grouplabels=false)
    F = h.args[1:end]

    seriestype := :scatter

    color_palette = [:cornflowerblue, :crimson, :forestgreen]
    color_palette --> color_palette
    for i = 1:length(h.args)
        @series begin
            markerstrokewidth --> 0
            label := nothing
            seriescolor --> color_palette[i]
            markeralpha --> exp(-0.002*size(F[i], 2))
            (x, y) = (F[i][1, :], F[i][2, :])
        end
        @series begin
            seriesalpha := ellipsealpha
            seriescolor := color_palette[i]
            seriestype := :shape
            if grouplabels isa Array
                label := grouplabels[i]
            else
                label := grouplabels
            end
            linewidth := 2
            linecolor := color_palette[i]
            linealpha := 1.0
            framestyle --> :box
            μ, S = StatsPlots._covellipse_args(Array.((mean(F[i], dims=2)[:], cov((F[i]), dims=2))); n_std=1.5)
            θ = range(0, 2π; length=1000)
            A = S * [cos.(θ)'; sin.(θ)']
            (μ[1] .+ A[1,:], μ[2] .+ A[2,:])
        end
    end
end