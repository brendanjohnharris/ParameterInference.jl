using WordCloud

WordCloud.wordcloud(𝑓::FeatureSet, args...; kwargs...) = wordcloud(string.(getnames(𝑓)), args...; kwargs...)
"""
We want to take a FeatureSet and and vector of weights for a PC and produce a word cloud
"""
function lowdimwordcloud(𝑓::FeatureSet, weights::AbstractVector; path="../Figures/$(rand(UInt16)).svg")
    @assert length(𝑓) == length(weights)
    wc = wordcloud(𝑓, weights; colors=:seaborn_dark, angles = 0, maskshape=squircle, outline=false, backgroundcolor = (1, 1, 1), font="Arial Medium", masksize=(1000, 400), density=0.1)
    generate!(wc)
    paint(wc, path)
    return wc
end

function lowdimwordcloud(𝑓::FeatureSet, M, PCs, args...; kwargs...)
    weights = sum(abs.(featureweights(M, PCs)), dims=2)[:]
    lowdimwordcloud(𝑓, weights, args...; kwargs...)
end

export lowdimwordcloud
