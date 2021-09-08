function baselinetransform(Fₕ, Fₗ=zeros(size(Fₕ)))
    # ! Temporary! Better to use RPCA!
    function transform(F)
        # Robust sigmoid normlise
        #normalise(F, median(F, dims=2), mapslices(iqr, F, dims=2), sigmoidNormalise, 2)
        # !TEMPORARY!
        for i ∈ 1:size(F, 1)
            σ = iqr(F[i, :])
            μ = median(F[i, :])
            for j ∈ 1:size(F, 2)
                if abs(F[i, j] - μ) > 5*σ
                    F[i, j] = μ
                end
            end
            if std(F[i, :]) > 1000
                F[i, :] .= 0.0
            end
        end
        return F
    end
    #transform = F -> normalise(F, median(Fₕ, dims=2), mapslices(iqr, Fₕ, dims=2), logistic, 2) # Robust sigmoid according to high dim
    Fₕ = Fₕ |> transform
    return F -> F |> transform #|> yeojohnson(Fₕ)
end
export baselinetransform
