function baselinefilter(Fₕ, Fₗ=zeros(size(Fₕ)); fitlervariance=true)
    initialsize = size(Fₕ, 1);
    nanfilter = _nonanrows(Fₕ);
    Fₕ, Fₗ = nanfilter.((Fₕ, Fₗ))

    inffilter = _noinfrows(Fₕ);
    Fₕ, Fₗ = inffilter.((Fₕ, Fₗ))

    integerfilter = _nointegerfeatures(Fₕ, 10, 3) # * Threshold at 10 unique values, to 3 significant figures.
    Fₕ, Fₗ = integerfilter.((Fₕ, Fₗ))

    function lowdimfilter(Fₕ, Fₗ)
        # * Get features that have a low dim variance smaller than 1% the high dim variance
        σₕ, σₗ = std.((Fₕ, Fₗ), dims=2)
        prop = σₗ./σₕ
        idxs = prop .< 0.01
        return F -> F[idxs, :]
    end
    variancefilter = filtervariance ? lowdimfilter(Fₕ, Fₗ) : identity

    @info "Filtered down to $(size(Fₕ, 1))/$initialsize features"
    return F -> F |> nanfilter |> inffilter |> integerfilter |> variancefilter # The order of these matters
end
export baselinefilter
