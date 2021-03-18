using Catch22
using ParameterInference
using UrlDownload
using CSV
using DataFrames
using Distances
using Plots
pyplot()
X = urldownload("https://ndownloader.figshare.com/files/24950795", true, format=:CSV, header=false, delim=",", type=Float64) |> Array;
#X = hcat([collect(x) for x in X]...); # time × time series matrix

🏷️ = urldownload("https://ndownloader.figshare.com/files/24950798", false, format=:CSV, header=true, delim=",") |> DataFrame;

X = [nomissing(collect(x)) for x in X];
F = hcat([catch22(Float64.(x)) for x in X]...);
F̂ = robustNormalise(F, logistic);
#Dt = pairwise(CorrDist(), F̂, dims=2)
#Df = pairwise(CorrDist(), F̂, dims=1) # mapslices(StatsBase.tiedrank, F̂, dims=2)
Df = 1.0.-abs.(StatsBase.corspearman(F̂'))
idxs = clusterDistances(Df)

# F̃ = clusterReorder(Array(F̂)', CorrDist(), linkageMetric=:average, branchOrder=:optimal)';
# p1 = plot(F̃, seriestype = :heatmap, seriescolor = cgrad(:RdYlBu_11, 7, categorical = true))

p2 = plot(Df[idxs, idxs], seriestype = :heatmap, aspect_ratio=:equal, xaxis=nothing)
plot!(yticks=(1:size(Df, 1), replace.(string.(Catch22.featureNames[idxs]), '_'=>"\\_")), size=(800, 400), xlims=[0.5, size(Df, 1)+0.5], ylims=[0.5, size(Df, 1)+0.5], box=:on, colorbar_title="1-|ρ|", clims=(0.0, 1.0))
