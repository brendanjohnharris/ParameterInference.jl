using StatsPlots

# ------------------------------------------------------------------------------------------------ #
#                 Compare the distributions of features in a feature matrices                      #
# ------------------------------------------------------------------------------------------------ #

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
        if F isa AbstractFeatureArray && F2 isa AbstractFeatureArray
            if size(F, 1) > size(F2, 1)
                (F2, F) = intersectFeatures(F2, F)
            else
                (F, F2) = intersectFeatures(F, F2)
            end
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
    elseif normtype == :none
        f1 = x -> 1.0
        f2 = x -> 0.0
    end
    if (normalise == :feature) && !isnothing(F2)
        S = mapslices(f1, hcat(F, F2), dims=2)
        C = mapslices(f2, hcat(F, F2), dims=2)
        S[S .== 0] .= 1.0
        F = (F.-C)./S
        F2 = (F2.-C)./S
        yaxis --> nothing
    elseif isnothing(F2) || normalise
        S = mapslices(f1, F, dims=2)
        C = mapslices(f2, F, dims=2)
        S[S .== 0] .= 1.0
        F = (F.-C)./S
        if ~isnothing(F2)
            F2 = (F2.- mapslices(x -> mean(x), F2, dims=2))./S
        end
    end

    if F isa AbstractFeatureArray && F2 isa AbstractFeatureArray
        fnames = replace.(String.(Catch22.featureDims(F)),  '_'=>"\\_")
    else
        fnames = 1:size(F, 1)
    end

    if backend() !== Plots.PyPlotBackend()
        rotation --> 90
        bottom_margin --> 10Plots.cm
    else
        tickfontrotation --> 90
    end
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
@recipe function f(P::VarianceMapping; interval=rampInterval, names=false)
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
    𝐟 = interval.(Fₗ, Fₕ, F)
    𝐟 = Catch22.featureVector(𝐟, Catch22.featureDims(Fₗ))
    (F, 𝐟) = intersectFeatures(F, 𝐟)

    xx = -0.1:0.01:1.1
    f₁ = interval(0.0, 1.0)
    yy = f₁.(xx)
    𝛔ᵣ = [𝐟[s, 1](F[s, :]) for s ∈ fnames]

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


@userplot IntervalScaling
@recipe function f(P::IntervalScaling; interval=rampInterval, reftoramp=true)
    if length(P.args) > 0 # It's evolving, but backwards
        σₗ = P.args[1]
    else
        σₗ = 0.0
    end
    if length(P.args) > 1
        σₕ = P.args[2]
    else
        σₕ = π
    end
    fl = rand(1, 1000).*σₗ
    fh = rand(1, 1000).*σₕ

    𝛔 = LinRange(σₗ, σₕ, 10000)
    f = [rand(1, 20).*𝛔[i] for i ∈ 1:length(𝛔)]
    x = StatsBase.std.(f)
    y = [interval(fl, fh, f[i])[1](f[i]) for i ∈ 1:length(x)]
    if reftoramp
        @series begin
            seriestype := :line
            label --> nothing#"Ramp Interval"
            linewidth --> 2.5
            seriescolor --> :black
            (x, [rampInterval(fl, fh, f[i])[1](f[i]) for i ∈ 1:length(x)])
        end
    end
    @series begin
        seriestype := :line
        seriescolor := crimson
        linewidth --> 2.5
        label --> nothing
        xguide --> "σ"
        yguide --> "σ̂"
        (x, y)
    end
end
