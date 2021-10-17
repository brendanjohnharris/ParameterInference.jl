using ParameterInference
using NonstationaryProcesses
using Plots
using Catch22
fourseas!()

# * Define a path in parameter space for an AR system
N = 1000000
a = sineWave((pa = (N, 0.6, 0, 0))...)
plot(0:N, a)
b = sineWave((pb = (N, 0.6, -N÷4, 0))...)
plot!(0:N, b)

S = arSim(parameter_profiles = (sineWave, sineWave),
          parameters = (pa, pb),
          tmax = N,
          transient_t0 = 0,
          t0 = 0,
          X0 = [0.0]) # Specifying 1 initial condition returns a univariate time series.


@userplot FeatureSpaceImage
@recipe function f(g::FeatureSpaceImage; var=1, lowdim=principalcomponents, nwindows=50, colors=[:cornflowerblue, :crimson])
    @assert g.args[1] isa Process
    S = g.args[1]
    if length(g.args) == 2
        f = g.args[2]
    else
        f = catch22
    end
    layout := (1, 2)
    p = parameterseries(S)
    t = times(S)
    x = timeseries(S, var)
    X, _ = slidingWindow(x, length(t)÷nwindows)
    F = f(X)
    if size(F, 1) > 2 # Do PCA
        m = F |> Array |> lowdim
        F = embed(m, F, 1:2)
    end
    aspect_ratio --> :equal
    arrow --> arrow(:closed, :head, 30.0, 20.0)
    size --> (800, 400)
    bottom_margin --> 0.5Plots.cm
    left_margin --> 0.5Plots.cm
    narrows = 4
    for nn = 1:narrows
        Nn = length(t)÷narrows
        if nn == narrows
            nx = Nn*(nn-1)+1:length(t)
        else
            nx = Nn*(nn-1)+1:Nn*nn
        end
        @series begin
            subplot := 1
            title := "Parameter space"
            seriescolor --> colors[1]
            titlefontsize --> 16
            xguide --> "\$\\theta_1\$"
            yguide --> "\$\\theta_2\$"
            linealpha --> 0.75
            seriestype --> :path
            markersize --> 2
            markercolor --> colors[1]
            x := p[1, nx]
            y := p[2, nx]
            xlims := [minimum(p[1, :])-0.2, maximum(p[1, :])+0.2]
            ylims := [minimum(p[2, :])-0.2, maximum(p[2, :])+0.2]
            ()
        end
        Nn = size(F, 2)÷narrows
        if nn == narrows
            nx = Nn*(nn-1)+1:size(F, 2)
        else
            nx = Nn*(nn-1)+1:Nn*nn+1
        end
        @series begin
            subplot := 2
            aspect_ratio := :auto
            seriescolor --> colors[2]
            seriestype --> :path
            linealpha --> 0.75
            x := F[1, nx]
            y := F[2, nx]
            ()
        end
    end
    @series begin
        subplot := 2
        aspect_ratio := :auto
        arrow := nothing
        title := "Feature space"
        seriescolor --> colors[2]
        titlefontsize --> 16
        xguide --> "\$f_1\$"
        yguide --> "\$f_2\$"
        markeralpha --> 0.75
        seriestype --> :scatter
        linealpha --> 0.5
        markersize --> 2.5
        markerstrokewidth --> 0.01
        x := F[1, :]
        y := F[2, :]
        xlims := [minimum(F[1, :])-0.2, maximum(F[1, :])+0.2]
        ylims := [minimum( F[2, :])-0.2, maximum( F[2, :])+0.2]
        ()
    end
end

featurespaceimage(S, AC[1:2], colors=[:black, :black]);
savefig("../Figures/$(rand(UInt16)).svg");
l = length(timeseries(S, 1))
n = round(Int, 2*8/3)
plot(vec(timeseries(S, 1))[n*l÷8+1:n*l÷8+251], color=:black);
savefig("../Figures/$(rand(UInt16)).svg");




SS = arSim(parameter_profiles = (constant,),
          parameters = (0.97),
          tmax = 250,
          transient_t0 = 0,
          t0 = 0,
          X0 = [0.0])
plot(SS)
savefig("../Figures/$(rand(UInt16)).svg");


SS = arSim(parameter_profiles = (constant,),
          parameters = (0.03),
          tmax = 250,
          transient_t0 = 0,
          t0 = 0,
          X0 = [0.0])
plot(SS)
savefig("../Figures/$(rand(UInt16)).svg");

SSS = updateparam(ikedaSim(), 4, 0.6)
plot(timeseries(SSS, 1)[1:150])
savefig("../Figures/$(rand(UInt16)).svg");

SSS = updateparam(ikedaSim(), 1, 25)
plot(timeseries(SSS, 1)[1:150])
savefig("../Figures/$(rand(UInt16)).svg");
