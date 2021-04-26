# ------------------------------------------------------------------------------------------------ #
#          Run a statistical test to compare the distributions of features in two matrices         #
# ------------------------------------------------------------------------------------------------ #
function statfeaturefilter(F₁, F₂; tail=:firstlarger, test=TailedTestOfVariance, alpha=0.05)

    p = testFeatureDistributions(F₁, F₂, test; tail=tail)
    issig = p .< alpha
    strs = deepcopy(p)
    strs[issig] = 10.0.^(ceil.(log10.(p[issig])))
    strs[.!issig] = round.(strs[.!issig], sigdigits=2)
    strs = string.(strs)
    strs[issig] = "<".*strs[issig]
    ann = [text("$(strs[i])", [:crimson, :black][issig[i]+1], 8, rotation=90, valign=:bottom) for i ∈ 1:length(p)] # :forestgreen
    plt = featureviolin(F₁, F₂, rightannotations=ann)
end
