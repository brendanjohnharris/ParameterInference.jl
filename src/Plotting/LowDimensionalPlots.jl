# ------------------------------------------------------------------------------------------------ #
#          Run a pca on the (time series of a) feature matrix and plot the first two PC's          #
# ------------------------------------------------------------------------------------------------ #
@userplot LowDim2
@recipe function f(h::LowDim2; method=principalComponents, classlabels=true, features=Catch22.featureDims(h.args[1]))
    # show(size(F))
    F = h.args[1]
    fs = Catch22.featureDims(F)
    seriestype := scatter
    M = project(standardise(Array(F)), method)
    F′ = embed(M, standardise(Array(F)), 1:2)

    if typeof(M) <: MultivariateStats.PCA
        PV = principalvars(M)
        PV = PV[1:2]./sum(PV)
        wmax = mapslices(x -> sum(abs.(x)), PCfeatureWeights(M)[:, 1:2], dims=1)
        weights, fis = findmax(PCfeatureWeights(M)[:, 1:2], dims=1)
        PV = [round(x*100, sigdigits=2) for x ∈ PV]
        w = [round(100*weights[i]/wmax[i], sigdigits=2) for i ∈ 1:length(weights)]
        #xguide --> "$(PV[1])% Variance ($(w[1])% $(fs[fis[1]]))"
        #yguide --> "$(PV[2])% Variance ($(w[2])% $(fs[fis[1]]))"
        xguide --> "PC1 ($(PV[1])%)"
        yguide --> "PC2 ($(PV[2])%)"
    end


    if typeof(classlabels) <: Bool && classlabels
        classlabels = dims(F)[2].val
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
    end
end



@userplot LowDim3
@recipe function f(h::LowDim3; method=principalComponents, classlabels=true, features=Catch22.featureDims(h.args[1]))
    # show(size(F))
    F = h.args[1]
    fs = Catch22.featureDims(F)
    seriestype := scatter
    M = project(standardise(Array(F)), method)
    F′ = embed(M, standardise(Array(F)), 1:3)

    markersize --> 2
    markerstrokewidth --> 0
    size --> (600, 400)
    if typeof(M) <: MultivariateStats.PCA
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


    if typeof(classlabels) <: Bool && classlabels
        classlabels = dims(F)[2].val
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
