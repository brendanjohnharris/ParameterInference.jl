@userplot DimensionalityEstimate
@recipe function f(D::DimensionalityEstimate)
    if D.args[1] isa Inference
        I = D.args[1]
        M = I.model
        F = Array(I.F̂)
    else
        F = Array(D.args[1])
        if length(D.args) > 2
            model = D.args[2]
        else
            model = principalcomponents
        end
        M = model(F)
    end
    σ² = residualVariance(M, F)
    ξ² = explainedVariance(M)

    legend --> false
    #link := :x
    bottom_margin --> 20Plots.mm
    left_margin --> 20Plots.mm
    grid --> false
    size --> (1000, 400)
    framestyle := :box
    layout --> @layout [ev rv]

    @series begin
        seriestype := :path
        markersize --> 5
        subplot := 1
        marker --> :circle
        label --> nothing
        seriescolor --> cornflowerblue
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
        seriescolor --> crimson
        xguide --> "Principal Components"
        yguide --> "Explained Variance"
        (x, y) = (0:length(ξ²), [0, ξ²...])
    end
end


@userplot OrthonormalisedBaseline
@recipe function f(D::OrthonormalisedBaseline)
    F = D.args[1]
    Fₕ = D.args[2]
    if D.args[3] isa AbstractArray
        Fₗ = D.args[3]
        interval = (x, y) -> NonstationaryProcesses.rampInterval(0, 1, x, y)
        𝑏 = orthonormalHiloBaseline(Fₗ, Fₕ; interval)
    else
        𝑏 = D.args[3]
    end

    legend --> false
    framestyle := :box

    ξ²ₕ = sort(explainedVariance(𝑏(Fₕ)), rev=true)
    ξ² = sort(explainedVariance(𝑏(F)), rev=true)

    @series begin
        seriestype := :path
        linestyle := :dash
        linewidth := 2.5
        label --> nothing
        seriescolor --> :gray
        (x, y) = (0:length(ξ²), LinRange(0, 1, length(ξ²)+1))
    end

    @series begin
        seriestype := :path
        markersize --> 5
        marker --> :circle
        label --> "Fₕ"
        seriescolor --> :black
        (x, y) = (0:length(ξ²ₕ), cumsum([0.0, ξ²ₕ...]))
    end

    @series begin
        seriestype := :path
        markersize --> 5
        marker --> :circle
        label --> "F"
        seriescolor --> :crimson
        xguide --> "Principal Components"
        yguide --> "ξ²"
        (x, y) = (0:length(ξ²), cumsum([0.0, ξ²...]))
    end
end

