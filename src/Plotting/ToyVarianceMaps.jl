using Distributions
using StatsPlots
using LinearAlgebra
# ------------------------------------------------------------------------------------------------ #
#                Plot some toy distributions to get a feel for the variance mappings               #
# ------------------------------------------------------------------------------------------------ #
@userplot MapDistributionVariance
@recipe function f(V::MapDistributionVariance; distributioncolors=[:black, :crimson, :cornflowerblue], ellipsealpha=0.2, doscatter=true)
    D = V.args[1:3] # Should be D, Dₗ, Dₕ
    if length(V.args) == 4
        b = V.args[4]
        size --> (900, 400)
        layout := @layout [x y]
    else
        b = nothing
    end
    framestyle --> :box
    labels = ["Test", "Low Dim.", "High Dim."]
    aspect_ratio --> :equal
    #axis --> nothing
    N = 200
    markersize --> 2
    legend --> (0.095, 0.96)
    legendfontsize --> 6
    markerstrokewidth --> 0
    # Plot the unscaled distributions
    for i ∈ [3, 1, 2]
        if doscatter
            @series begin
                subplot := 1
                seriescolor := distributioncolors[i]
                label := nothing
                seriestype := :scatter
                X = rand(D[i], N)
                (X[1, :], X[2, :])
            end
        end
        @series begin
            title --> "Raw"
            subplot := 1
            seriesalpha := ellipsealpha
            seriescolor := distributioncolors[i]
            label := labels[i]
            seriestype := :shape
            linewidth := 2
            linecolor := distributioncolors[i]
            linealpha := 1.0
            𝛉 = range(0, 2π; length=100) # From StatsPlots CovEllipse
            #A = cov(D[i]) * [cos.(θ)'; sin.(θ)'] # This COVARIANCE, which is a bit pointless in data space
            # With a bit of rearranging we can get an SD shape, which will scale with the data
            r = [sqrt.([cos(θ); sin(θ)]'*cov(D[i])*[cos(θ); sin(θ)]) for θ ∈ 𝛉]
            x = r.*cos.(𝛉)
            y = r.*sin.(𝛉)
            (x, y)
        end
    end

    # Plot the baseline scaled distributions
    if !isnothing(b)
        for i ∈ [3, 1, 2]
            v = [b[s](sqrt(var(D[i])[s]))./sqrt(var(D[i])[s]) for s ∈ 1:2]
            if doscatter
                @series begin
                    subplot := 2
                    seriescolor := distributioncolors[i]
                    label := nothing
                    seriestype := :scatter
                    X = rand(D[i], N)
                    (X[1, :].*v[1], X[2, :].*v[2])
                end
            end
            @series begin
                title --> "Scaled"
                subplot := 2
                seriesalpha := ellipsealpha
                seriescolor := distributioncolors[i]
                label := labels[i]
                seriestype := :shape
                linewidth := 2
                linecolor := distributioncolors[i]
                linealpha := 1.0
                𝛉 = range(0, 2π; length=100)
                r = [sqrt.([cos(θ); sin(θ)]'*cov(D[i])*[cos(θ); sin(θ)]) for θ ∈ 𝛉]
                x = r.*cos.(𝛉)
                y = r.*sin.(𝛉)
                (x.*v[1], y.*v[2])
            end
        end
    end

end