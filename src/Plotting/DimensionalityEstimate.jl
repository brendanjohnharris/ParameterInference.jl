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


@userplot OrthonormalisedBaseline
@recipe function f(D::OrthonormalisedBaseline)
    F = D.args[1]
    Fₕ = D.args[2]
    if typeof(D.args[3]) <: AbstractArray
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
        markersize --> 5
        subplot := 1
        marker --> :circle
        label --> "Fₕ"
        seriescolor --> :black
        (x, y) = (0:length(ξ²ₕ), cumsum([0.0, ξ²ₕ...]))
    end

    @series begin
        seriestype := :path
        markersize --> 5
        subplot := 1
        marker --> :circle
        label --> "F"
        seriescolor --> :red
        xguide --> "Principal Components"
        yguide --> "ξ²"
        (x, y) = (0:length(ξ²), cumsum([0.0, ξ²...]))
    end
end

