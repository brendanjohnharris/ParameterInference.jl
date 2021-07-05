using Plots


# Window a time series and plot a feature
@shorthands featurepath
@recipe function f(::Type{Val{:featurepath}}, x, y, z; features=catch24, windows=slidingWindow(length(x)÷20))
    # x is time, y is time series
    X, t = windows(y)
    t = x[t]
    F = features(X)

    seriestype := :path
    framestyle --> :box
    linewidth --> 2.5
    size --> (1200, 800)
    legend --> :outerright
    xguide --> "Time"
    yguide --> "Feature"
    for f ∈ getnames(features)
        @series begin
            label := string(f)
            x := t[1:end-1].+diff(t)./2
            y := (size(F, 1)==1 ? F[:] : vec(F[f, :]))
        end
    end
end
