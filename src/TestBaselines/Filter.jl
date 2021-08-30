function baselinefilter(Fₕ, Fₗ=zeros(size(Fₕ)))

    nanfilter = _nonanrows(Fₕ);
    Fₕ, Fₗ = nanfilter.((Fₕ, Fₗ))

    inffilter = _noinfrows(Fₕ);
    Fₕ, Fₗ = inffilter.((Fₕ, Fₗ))

    integerfilter = _nointegerfeatures(Fₕ, 5, 3) # * Threshold at 5 unique values, to 3 significant figures.
    Fₕ, Fₗ = integerfilter.((Fₕ, Fₗ))

    return F -> F |> nanfilter |> inffilter |> integerfilter # The order of these matters
end
