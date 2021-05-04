using StatsPlots

# ------------------------------------------------------------------------------------------------ #
#                 Compare the distributions of features in a feature matrices                      #
# ------------------------------------------------------------------------------------------------ #

# I NEED a custom FeatureArray
@userplot FeatureViolin
@recipe function f(H::FeatureViolin; normalise=:feature, normtype=:mix, rightannotations=nothing)
    # if normalise is on, normalise distributions to the range of the first feature matrix but centre uniquely
    F = H.args
    if length(F) == 1
        F = F[1]
        F[isinf.(F)] .= 0
        F2 = nothing
    elseif length(F) == 2
        F, F2 = F
        F[isinf.(F)] .= 0
        F2[isinf.(F2)] .= 0
        if size(F, 1) > size(F2, 1)
            (F2, F) = intersectFeatures(F2, F)
        else
            (F, F2) = intersectFeatures(F, F2)
        end
    end
    if normtype == :zscore
        f1 = x -> StatsBase.std(x)
        f2 = x -> mean(x)
    elseif normtype == :unit
        f1 = x -> abs(-(extrema(x)...))
        f2 = x -> mean(extrema(x))
    elseif normtype == :mix
        f1 = x -> abs(-(extrema(x)...))
        f2 = x -> mean(x)
    end
    if normalise == :feature && !isnothing(F2)
        S = mapslices(f1, hcat(F, F2), dims=2)
        C = mapslices(f2, hcat(F, F2), dims=2)
        S[S .== 0] .= 1.0
        F = (F.-C)./S
        F2 = (F2.-C)./S
        yaxis --> nothing
    elseif normalise
        S = mapslices(f1, F, dims=2)
        C = mapslices(f2, F, dims=2)
        S[S .== 0] .= 1.0
        F = (F.-C)./S
        if ~isnothing(F2)
            F2 = (F2.- mapslices(x -> mean(x), F2, dims=2))./S
        end
    end

    fnames = replace.(String.(Catch22.featureDims(F)),  '_'=>"\\_")

    tickfontrotation --> 90
    size --> (700, 900)
    legend := false

    if ~isnothing(F2)
        @series begin
            seriestype := :violin
            side := :right
            seriescolor := :crimson
            ((1:size(F2, 1))', Array(F2)')
        end
    end


    @series begin
        seriestype := :violin
        if ~isnothing(rightannotations)
            top_margin --> 40Plots.mm
            annotations := [(i, 1.1, rightannotations[i]) for i ∈ 1:length(rightannotations)]
        end
        #bottom_margin --> 70Plots.mm
        #top_margin --> 70Plots.mm
        tickfontvalign := :top
        framestyle --> :box
        ymirror --> true
        if ~isnothing(F2); side := :left; seriescolor := :cornflowerblue; end
        xticks --> (1:size(F, 1), fnames)
        ((1:size(F, 1))', Array(F)')
    end

end





# ------------------------------------------------------------------------------------------------ #
#                                Plot the baseline variance mapping                                #
# ------------------------------------------------------------------------------------------------ #
@userplot VarianceMapping
@recipe function f(P::VarianceMapping; interval=(x, y) -> NonstationaryProcesses.rampInterval(0, 1, x, y), names=false)
    F = P.args[1]
    Fₗ = P.args[2]
    Fₕ = P.args[3]
    legend --> false
    #link := :x
    grid --> true
    yticks --> [0.0, 1.0]
    xguide --> "Input Variance"
    yguide --> "Output Variance"
    left_margin --> 10Plots.mm

    if any(Catch22.featureDims(Fₗ) .!= Catch22.featureDims(Fₕ))
        error("High and low dimensional baselines do not have the same features")
    end

    𝛔ₗ, 𝛔ₕ = std(Fₗ, dims=2), std(Fₕ, dims=2)
    𝛔ₕ[𝛔ₕ .< 𝛔ₗ] .= Inf
    fnames = Catch22.featureDims(F)
    𝐟 = interval.(vec(𝛔ₗ), vec(𝛔ₕ))
    𝐟 = Catch22.featureVector(𝐟, Catch22.featureDims(Fₗ))
    (F, 𝐟) = intersectFeatures(F, 𝐟)
    𝛔 = std(F, dims=2)


    xx = -0.1:0.01:1.1
    f₁ = interval(0.0, 1.0)
    yy = f₁.(xx)
    𝛔ᵣ = [𝐟[s, 1](𝛔[s, 1]) for s ∈ fnames]
    @series begin
        seriescolor --> :black
        linewidth --> 3
        (xx, yy)
    end

    @series begin
        #clims := extrema(vec(𝛔ᵣ))
        marker_z --> vec(𝛔ᵣ)
        markersize --> 10
        tickfontsize --> 12
        seriescolor --> cgrad(:RdGy_4, rev=true)
        seriestype --> :scatter
        xticks --> ((0.0, 1.0), ("σₗ", "σₕ"))
        (d, 𝛔ₗ′) = intersectFeatures(𝛔, 𝛔ₗ)
        (d, 𝛔ₕ′) = intersectFeatures(𝛔, 𝛔ₕ)
        x = ((vec(𝛔) .- vec(𝛔ₗ′))./(vec(𝛔ₕ′) .- vec(𝛔ₗ′)))
        y = vec(𝛔ᵣ)
        if names
            ftsz = [1+Int(round(y[i]./max(y...))).*8 for i ∈ 1:lastindex(fnames)]
            annotations := [(x[i]+0.05, y[i], text("$(fnames[i])", :black, :left, ftsz[i])) for i ∈ 1:lastindex(fnames)]
        end
        (x, y)
    end


end
