function baselinefilter(Fₕ, Fₗ=zeros(size(Fₕ)))
    initialsize = size(Fₕ, 1);
    nanfilter = _nonanrows(Fₕ);
    Fₕ, Fₗ = nanfilter.((Fₕ, Fₗ))

    inffilter = _noinfrows(Fₕ);
    Fₕ, Fₗ = inffilter.((Fₕ, Fₗ))

    integerfilter = _nointegerfeatures(Fₕ, 10, 3) # * Threshold at 10 unique values, to 3 significant figures.
    Fₕ, Fₗ = integerfilter.((Fₕ, Fₗ))
    @info "Filtered down to $(size(Fₕ, 1))/$initialsize features"
    return F -> F |> nanfilter |> inffilter |> integerfilter # The order of these matters
end
export baselinefilter
