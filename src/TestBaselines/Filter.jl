function baselinefilter(Fₕ, Fₗ=zeros(size(Fₕ)); filtervariance=true, thresh=0.05)
    initialsize = size(Fₕ, 1);

    nanfilter = _nonanrows(Fₕ);
    Fₕ, Fₗ = nanfilter.((Fₕ, Fₗ))

    inffilter = _noinfrows(Fₕ);
    Fₕ, Fₗ = inffilter.((Fₕ, Fₗ))

    integerfilter = _nointegerfeatures(Fₕ, 10, 3) # * Threshold at 10 unique values, to 3 significant figures.
    Fₕ, Fₗ = integerfilter.((Fₕ, Fₗ))

    function lowdimfilter(Fₕ, Fₗ)
        # * Get features that have a low dim variance smaller than thresh% the high dim variance
        σₕ, σₗ = std.((Fₕ, Fₗ), dims=2)
        prop = Array(σₗ)./Array(σₕ)
        idxs = vec(prop .< thresh)
        return F -> ((@warn "Filtered by variances down to $(sum(idxs))/$(length(idxs)) features"),
                    F[idxs, :])[2]
    end
    variancefilter = filtervariance ? lowdimfilter(Fₕ, Fₗ) : identity

    @warn "Filtered down to $(size(Fₕ, 1))/$initialsize features"
    return F -> F |> nanfilter |> inffilter |> integerfilter |> variancefilter # The order of these matters
end
export baselinefilter

function baselinefilter(; kwargs...)
    return (x, y) -> baselinefilter(x, y; kwargs...)
end
