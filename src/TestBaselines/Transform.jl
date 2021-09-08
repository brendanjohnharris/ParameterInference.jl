function baselinetransform(Fₕ, Fₗ=zeros(size(Fₕ)))
    return F -> F |> yeojohnson(Fₕ)
end
export baselinetransform
