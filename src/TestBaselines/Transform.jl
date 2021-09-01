function baselinetransform(Fₕ, Fₗ=zeros(size(Fₕ)))

    return F -> F
end
export transform
