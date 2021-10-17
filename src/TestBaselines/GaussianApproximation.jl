### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 09bee220-1249-11ec-3e2a-4fdde9976a59
begin
	import Pkg
    Pkg.activate(mktempdir())
	Pkg.add(url="https://github.com/brendanjohnharris/Catch22.jl#main")
	using Catch22
	Pkg.add(url="https://github.com/tk3369/YeoJohnsonTrans.jl")
	Pkg.develop(path="C:\\Users\\Brendan\\OneDrive - The University of Sydney (Students)\\Honours\\Code\\ParameterInference.jl")
	Pkg.add(path="C:\\Users\\Brendan\\OneDrive - The University of Sydney (Students)\\Honours\\Code\\ParameterInference.jl")
	using ParameterInference, Plots, StatsBase, Statistics, LinearAlgebra, ParameterInference.NonstationaryProcesses, MultivariateStats
	gr(); fourseas!(); # Plots initialisation
	nothing 
end

# ╔═╡ 1cbf17f7-8322-4e3c-a895-163a48332205
md"# Baseline PCA"

# ╔═╡ 2f0cb4e5-4424-42d9-8eb7-e8bf128575ad
md"""
We will construct a dataset that demonstrates the impact of dependencies between features on PCA estimates.
"""

# ╔═╡ 0f10dc15-9153-4d8c-a5e5-c0318c459fa1
md"## Parameters"

# ╔═╡ 230c66e5-7898-48af-9544-439677bee137
md"For a simple example, choose to have two 'parameters' that drive our manually constructed 'features': 𝛼 and 𝛽. Both will vary as random Gaussians.:"

# ╔═╡ fee5a67f-1678-45fc-83b3-3ffc187c19b3
nₛ = 1000;

# ╔═╡ c3e707a4-1e35-4fe0-8e6d-88d0eca27fde
𝜂 = 0.25;

# ╔═╡ 5146ae95-d551-49ec-888a-71fc7bf65d14
𝛼, 𝛽 = randn(nₛ, 2) |> eachcol;

# ╔═╡ b646056a-7c34-4cb3-aac9-d255646f6b75
plotcov(x) = covarianceimage(1:5, x, colormode=:raw, docluster=false, colorbar_title="", title="\$|\\Sigma^2|\$");

# ╔═╡ 2fbff3c4-9377-4624-90d6-d4ea7c8da64e
md"# Feature scenario #1"

# ╔═╡ 404bb5c9-74ab-43a9-8791-67ac23b0c0f8
md"""Next, we construct 5 features according to the parameters. 
In our first scenario, these are:

| Feature      	| Description  	|
| ----------- 	| ----------- 	|
| 𝑓₁ -- 𝑓₄      | Sensitive to 𝛼, but intercorrelated  |
| 𝑓₅   		 | Sensitive to 𝛽  |
"""

# ╔═╡ 4f01861c-dd52-462f-b340-ceecd3385c8e
md"""### High-dimensional baseline
In which both 𝛼 and 𝛽 are varying:
"""

# ╔═╡ 92670bd1-ec33-4590-9a53-ceca6f25ac0f
𝑓ₕ₁ = hcat(𝛼, 𝛼, 𝛼, 𝛼, 𝛽)';

# ╔═╡ 8629c505-a1e4-4c73-b2d4-44ecc858373d
plotcov(𝑓ₕ₁)

# ╔═╡ c3c3d6c3-49a0-40a6-b41b-dbdfa20a7d6b
md"""### Test data
In which 𝛼 is constrained:
"""

# ╔═╡ 9247ba48-f242-4479-a92d-e994b1610279
𝑓₁ = hcat(𝜂.*[𝛼, 𝛼, 𝛼, 𝛼]..., 𝛽)';

# ╔═╡ c973721b-c20b-4573-a8fa-a9a9282fc87d
plotcov(𝑓₁)

# ╔═╡ 4032a036-842b-462b-b75d-ebaaaf4aded9
md"### True parameters"

# ╔═╡ 6a957b7d-8fff-46c5-b441-01f8b9d45eb7
scatter(𝛽, 𝜂.*𝛼, aspect_ratio=:equal, ylabel="𝛼", xlabel="𝛽", title="True parameter variation",  markersize=2, markercolor=:gray)

# ╔═╡ 5686638b-ba3f-4691-9291-d91dfa510aa7
md"### PCA"

# ╔═╡ 76e702f7-521c-4445-aa6c-de28215d8d44
𝛴² = cov(𝑓₁');

# ╔═╡ b17a0c26-ce25-4e11-84dd-0d4b86371ae5
𝑃 = eigvecs(𝛴²);

# ╔═╡ 69e0799e-18f4-4f19-9bae-44202c2bba5d
𝑓′₁ = 𝑃'*𝑓₁;

# ╔═╡ cf454ccf-e505-4df9-a3cf-4afba2211d4a
scatter(𝑓′₁[5, :], 𝑓′₁[4, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="PCA estimate"); # Scatter significant PCs

# ╔═╡ 8013d3a2-a011-456b-9bfe-3ec781c31aba
scatter!(𝛽, 𝜂.*𝛼, aspect_ratio=:equal, markersize=2, markercolor=:gray)

# ╔═╡ ce024bfd-b892-4a31-a0d1-1d4befc7b80e
md"### Standardised PCA"

# ╔═╡ 542aec21-e623-4e53-ae4b-a7f240bfcb08
𝑓̂₁ = 𝑓₁./std(𝑓₁, dims=2);

# ╔═╡ 1daa139a-23c0-4995-bf73-abfbb7c518dd
𝛴̂² = cov(𝑓̂₁');

# ╔═╡ 705068d7-033f-49d2-bb59-4b40aa514642
𝑃̂ =  fit(MultivariateStats.PCA, 𝑓̂₁).proj; #eigvecs(𝛴̂²);

# ╔═╡ f1a685e5-ab99-4a61-a2d9-41ebf33372b0
𝑓̂′₁ = 𝑃̂'*𝑓₁;

# ╔═╡ 3a7af0db-8e3d-4100-a7c2-dcd37253a9ce
scatter(𝑓̂′₁[2, :], 𝑓̂′₁[1, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Standardised PCA estimate");

# ╔═╡ 7a28c225-bab8-40ee-906e-0c5349accc84
scatter!(𝛽, 𝜂.*𝛼, aspect_ratio=:equal, markersize=2, markercolor=:gray)

# ╔═╡ 4d619965-6eb7-4528-804c-00bbf1572e60
md"""### Baseline PCA
In this case, whitening:"""

# ╔═╡ 7ddd5530-7590-42a7-8e66-54432f717eec
𝛴²ₕ = cov(𝑓ₕ₁');

# ╔═╡ a95a3d4c-cb8b-4840-bc67-57fb2be00065
m1 = fit(MultivariateStats.PCA, 𝑓ₕ₁; maxoutdim=5, pratio=1.0)

# ╔═╡ 688dc093-c173-4eb4-ad12-4ff2e899b9f9
𝑃ₕ = m1.proj; # eigvecs(𝛴²ₕ)

# ╔═╡ f4081452-9314-47a4-948d-2813bda53507
𝛬ₕ = sqrt(diagm(m1.prinvars)); # Diagonal(sqrt.(abs.(eigvals(𝛴²ₕ))));

# ╔═╡ 5b853331-88c7-4b7f-b26e-8e0f963a5ed5
𝑓′′₁ = (𝑃ₕ*inv(𝛬ₕ))'*𝑓₁;

# ╔═╡ 1ee56732-671e-447e-adb8-cce45902b935
scatter(𝑓′′₁[2, :], .-𝑓′′₁[1, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′"); 

# ╔═╡ 70d2fe5f-7f7d-4891-8490-5c508c21c6f6
scatter!(𝛽, 𝜂.*𝛼, aspect_ratio=:equal, markersize=2, markercolor=:gray)

# ╔═╡ bb52a2b3-e856-4785-abb9-78ba15382d11
md"# Feature scenario #2"

# ╔═╡ 70d2f196-1604-4cf4-8e38-05dd8c099c83
md"""What if features one to four are sensitive to both 𝛼 and 𝛽? Then the effect of constraining 𝛼 manifests as an increase in correlation between features. For our second scenario, the features are then:

| Feature      	| Description  	|
| ----------- 	| ----------- 	|
| 𝑓₁ -- 𝑓₄      | Sensitive to 𝛼 and 𝛽, but intercorrelated  |
| 𝑓₅   		 | Sensitive to 𝛽  |
"""

# ╔═╡ 009c662a-9940-43ef-b0c9-d55560e6f49b
md"### High-dimensional baseline"

# ╔═╡ 46e7d2a0-8b62-4112-92df-74538f5f16d8
𝑓ₕ₂ = hcat(𝛼+𝛽/4, 𝛼+𝛽/4, 𝛼+𝛽/4, 𝛼+𝛽/4, 𝛽)';

# ╔═╡ 595e5022-9099-45ab-b5b0-b92fedb96ba8
plotcov(𝑓ₕ₂./std(𝑓ₕ₂, dims=2))

# ╔═╡ 0d062a65-adca-42c4-8b1c-84095d5335c5
md"### Test data"

# ╔═╡ 0f6b9065-55ad-4b1e-bcc3-2e177a0e331d
𝑓₂ = hcat((𝜂.*[𝛼, 𝛼, 𝛼, 𝛼] .+ [𝛽, 𝛽, 𝛽, 𝛽]./4)..., 𝛽)';

# ╔═╡ fd506ba8-9a5b-4235-bc43-b730522dce4f
plotcov(𝑓₂./std(𝑓₂, dims=2)) 

# ╔═╡ 58083319-d649-4c2d-b4b2-c2df7c49ee46
md"### True parameters"

# ╔═╡ 2a9a4929-8755-4237-a454-fda125d548ca
scatter(𝛽, 𝜂.*𝛼 .+ 𝛽./4, aspect_ratio=:equal, ylabel="𝛼", xlabel="𝛽", title="True parameter variation",  markersize=2, markercolor=:gray)

# ╔═╡ 1762e05f-7eec-4411-865a-eba47531a49c
md"### Standardised PCA"

# ╔═╡ feeaa835-28e0-4a24-905b-53347560ec9a
𝑓̂₂ = 𝑓₂./std(𝑓₂, dims=2);

# ╔═╡ 2358f79f-f44d-4e93-b4da-75e7be715882
𝛴̂²₂ = cov(𝑓̂₂');

# ╔═╡ d9fca85a-63b6-4ab5-ab0e-6cd5e68c4e06
𝑃̂₂ = fit(MultivariateStats.PCA, 𝑓̂₂).proj; # eigvecs(𝛴̂²₂);

# ╔═╡ d700b4ae-8bd5-4437-b515-c21d3da70543
𝑓̂′₂ = 𝑃̂₂'*𝑓₂;

# ╔═╡ 06e226a6-0f0f-4ef6-b916-04d51a3789c8
scatter(𝑓̂′₂[1, :], .-𝑓̂′₂[2, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Standardised PCA estimate");

# ╔═╡ ea513127-e9bb-40a7-9ec0-2beee8fa6179
scatter!(𝛽, 𝜂.*𝛼 .+ 𝛽./4, markersize=2, markercolor=:gray)

# ╔═╡ f0a7f529-19ac-44f7-8ab1-8d72d596a0ef
md"""### Baseline PCA"""

# ╔═╡ f946e3d7-15b9-45a9-951b-b3d77fc589f8
𝛴²ₕ₂ = cov(𝑓ₕ₂');

# ╔═╡ 652b998d-a6b0-4bb7-908e-c3dc7a82764c
m2 = fit(MultivariateStats.PCA, 𝑓ₕ₂);

# ╔═╡ d192149c-1ef4-4761-9e78-879a97abee57
𝑃ₕ₂ = m2.proj; #eigvecs(𝛴²ₕ₂);

# ╔═╡ 14d5e08c-0808-4352-835e-c173c53fd8d7
𝛬ₕ₂ = sqrt(diagm(m2.prinvars)); # Diagonal(sqrt.(abs.(eigvals(𝛴²ₕ₂))))

# ╔═╡ 772d260a-72b3-4c9e-8e65-6368508cfb52
𝑓′′₂ = (𝑃ₕ₂*inv(𝛬ₕ₂))'*𝑓₂;

# ╔═╡ 0f83a03c-890d-4771-b83f-16d8d5db81e9
cov(𝑓′′₂,dims=2)

# ╔═╡ b9a8e7e4-5f9b-462d-845e-b873f50bfc90
scatter(𝑓′′₂[2, :], .-𝑓′′₂[1, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′");

# ╔═╡ f1df27ea-0537-4f4b-afe7-c0692ca7b26d
scatter!(𝛽, 𝜂.*𝛼 .+ 𝛽./4, markersize=2, markercolor=:gray)

# ╔═╡ 2f987bde-a87a-43a8-8653-0c5812c56a70
md"# Feature scenario #3"

# ╔═╡ 1d18e8c7-7eff-4b75-ad67-e61bc22e12ea
md"""What if each feature has some intrinsic noise? This requires incorporating low-dimensional variance into the estimate.
To show this:

| Feature      	| Description  	|
| ----------- 	| ----------- 	|
| 𝑓₁ -- 𝑓₃      | Sensitive to 𝛼, but intercorrelated and with small noise |
| 𝑓₄   		 | Sensitive to 𝛽, but with small noise  |
| 𝑓₅   		 | Small noise only  |
"""

# ╔═╡ 3821aa68-e3b3-4764-a7ff-0b6b0e6bdc3e
𝜂₃ = 0.25;

# ╔═╡ 9d0c3f2b-8cf4-41f9-8fe8-5cd9a19dbd88
𝑓ₕ₃ = hcat(repeat([𝛼.+𝜂₃.*randn(nₛ)],3)..., 𝛽.+𝜂₃.*randn(nₛ), (𝜂₃.-0.01).*randn(nₛ))';

# ╔═╡ b0d86eb8-6f94-424a-adb2-eae8e40be1fd
plotcov(𝑓ₕ₃)

# ╔═╡ e507beed-984e-4f8b-a59b-bb3dcff1416a
md"""### Zero-dimensional baseline
In which no parameters are varying:
"""

# ╔═╡ ea23a142-1c36-4430-a598-2fcf0f2e3b66
𝑓₀ = hcat(repeat([𝜂₃.*randn(nₛ)], 3)..., 𝜂₃.*randn(nₛ), 𝜂₃.*randn(nₛ))';

# ╔═╡ fe0902aa-b55e-4b72-a35e-77db673747c9
plotcov(𝑓₀)

# ╔═╡ 9b92548e-ae54-48de-a5c9-4be88ae11228
md"### Test data"

# ╔═╡ 33167762-e2c6-4aba-98b7-58e28ee35ccb
𝑓₃ = hcat(repeat([0.1.*𝛼 .+ 𝜂₃.*randn(nₛ)], 3)..., 𝛽.+𝜂₃.*randn(nₛ), 𝜂₃.*randn(nₛ))'; 

# ╔═╡ 35dcd411-a342-4ee2-9146-f45ad39d6341
plotcov(𝑓₃) 

# ╔═╡ d2404839-53bf-49d3-a9bf-a0ea81471c43
md"### True parameters"

# ╔═╡ e64a04de-9257-4bbb-84c4-007c532d9c72
scatter(𝛽, 0.1.*𝛼, aspect_ratio=:equal, ylabel="𝛼", xlabel="𝛽", title="True parameter variation",  markersize=2, markercolor=:gray)

# ╔═╡ 710d51ea-8196-4a7f-91fe-1a9bd3f06d01
md"""
### Standardised PCA
"""

# ╔═╡ a4c8677f-6bdc-4bfe-adcc-3130f45b7aaa
𝑓̂₃ = 𝑓₃./std(𝑓₃, dims=2);

# ╔═╡ 5ad01546-7daa-441f-bf2c-60e2a6e6f0c7
𝛴̂²₃ = cov(𝑓̂₃');

# ╔═╡ 36039419-6df3-45f2-ae2c-77cf6e3b405d
𝑃̂₃ = eigvecs(𝛴̂²₃);

# ╔═╡ d046eb57-d4cb-4eda-9a81-00a234af1d57
𝑓̂′₃ = 𝑃̂₃'*𝑓₃;

# ╔═╡ 4e9b0004-da7a-4cb3-afe9-c3365d37471a
scatter(𝑓̂′₃[4, :], 𝑓̂′₃[5, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Standardised PCA estimate");

# ╔═╡ 5e2281e9-360b-41dd-9f4f-38e14d8aea86
scatter!(𝛽, 0.1.*𝛼, markersize=2, markercolor=:gray)

# ╔═╡ fa43e761-0b22-4903-aa6c-47050d8c27c0
scatter(𝛽, 𝑓̂′₃[4, :]);

# ╔═╡ bcd85e11-8bf0-4968-a528-8b56ce14db5b
md"""### Baseline PCA"""

# ╔═╡ 3fe79e11-947e-46e6-a5ef-1bf1b537c857
md"""
The full baseline-corrected covariance is:

$$\Sigma^{\prime 2} = (P \Lambda_h^{-1})^T \left(\Sigma^2 - \Sigma_0^2\right)(P \Lambda_h^{-1}).$$

But this is not a linear transformation of feature space, so some approximations must (?) be made. 

An easy one is to treat the zero-dimensional covariance as diagonal. In that case:

$$\Sigma^2 - \Sigma_0^2 \approx V^T \Sigma^2 V,$$
where:

$$V = \sqrt{I - D(\Sigma^2_0) D(\Sigma^{-2})},$$
And $D(A)$ is the diagonal of $A$. The resulting approximation is simply a linear rescaling. Importantly, it satisfies the condition that $V \to 0$ as $\Sigma^2 \to \Sigma^2_0$.

There are other approximations that also allow covariances to vanish in the same limit...
"""

# ╔═╡ f771887a-77f0-4a4d-9e4c-7e988201197f
𝛴²ₕ₃ = cov(𝑓ₕ₃');

# ╔═╡ b636df98-f935-46b4-bb08-be96cee18884
𝛴²₀ = cov(𝑓₀')

# ╔═╡ c43d6053-4eeb-450a-bca4-bd5d58b179c1
𝛴²₃ = cov(𝑓₃')

# ╔═╡ fd00b9e4-f2fc-4571-8596-b1e53f8d7286
𝑃ₕ₃ = eigvecs(𝛴²ₕ₃);

# ╔═╡ 9d13988d-e625-42b5-8e89-86d519027912
𝛬ₕ₃ = Diagonal(sqrt.(abs.(eigvals(𝛴²ₕ₃))))

# ╔═╡ cb77a71b-7cba-489d-a51b-690cdd281a4c
𝑉 = sqrt(I - Diagonal(𝛴²₀)*inv(Matrix(Diagonal(𝛴²₃))));

# ╔═╡ e26f0ae5-4b0b-4886-a0ad-e109dbf9e321
𝑉'*𝛴²₃*𝑉;

# ╔═╡ dc9c5d21-87fc-404b-b8b7-1a704d890591
𝛴²₃ - 𝛴²₀;

# ╔═╡ 3eda9fc1-0d54-4cde-94ba-5e611a4773b6
𝛬ₕ₃[2, 2] = eps();

# ╔═╡ 0ae23619-0ce3-4e32-9a11-f9a444235c70
𝑓′′₃ = (𝑃ₕ₃*inv(𝛬ₕ₃)*𝑉)'*𝑓₃

# ╔═╡ f64c9529-1f71-4daa-8f4e-6af000288b38
𝛬ₕ₃

# ╔═╡ c3067e6d-dfaf-4938-8bbf-4b51514fd286
𝑓′′₀ = (𝑃ₕ₃*inv(𝛬ₕ₃)*𝑉)'*𝛴²₀*(𝑃ₕ₃*inv(𝛬ₕ₃)*𝑉); # Not 0 because the transform is a strong approximation

# ╔═╡ a2f4c245-031b-4f1c-9f44-b3947cbc9795
cov(𝑓′′₃, dims=2)

# ╔═╡ ad7aac3b-303a-4ca6-a44f-f11cba8d115f
𝑓′′₀

# ╔═╡ dbdcd6c4-102d-41cc-b218-100f74355b85
scatter(𝑓′′₃[4, :], .-𝑓′′₃[5, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Baseline PCA estimate", left_margin=5Plots.mm); 

# ╔═╡ 9d6334c1-7107-4fbb-b383-9a37fc56de74
scatter!(.-𝛽, .-0.1.*𝛼, markersize=1, markercolor=:gray)

# ╔═╡ 07e9a417-5adb-4182-a98d-fd618feaf675
scatter(𝛽, 𝑓′′₃[4, :]); # Compairwise to true parameters

# ╔═╡ 2826e49a-5566-4aa7-be9a-73799dfbebc7
scatter(𝑓̂′₃[4, :], 𝑓′′₃[4, :]); # Compare to standardised PCA estimate. Similar, which is good, but properly scaled relative to other PCs (as above).

# ╔═╡ c79d0914-d0db-4f5c-b1f2-a6154d063e78


# ╔═╡ eaae76e1-9326-432a-8fc5-b03c3d93e75a


# ╔═╡ b7a9e831-6ab4-4afd-a5da-9d3b5414245b
fₕ₃ = 𝑓ₕ₃ .+eps().*randn(size(𝑓ₕ₃))

# ╔═╡ 70109d54-cfcd-46c5-8c67-a3b96286ee09
f₃ = 𝑓₃ .+eps().*randn(size(𝑓₃))

# ╔═╡ 9970c6e2-a2c9-442e-8fca-17b8a5c360b9
𝑃ₕ₄ = eigvecs(𝛴²ₕ₃)

# ╔═╡ 7408597f-8bc6-493a-80f6-79a5201d625b
#Λ = Diagonal(sqrt.(abs.(eigvals(𝛴²ₕ₃))))

# ╔═╡ a031cdd3-7e92-45d8-8037-e6695cd588c2
MultivariateStats.PCA

# ╔═╡ 816e841b-3f3b-4b8c-8c4b-4cd93c685ab7
m = fit(MultivariateStats.PCA, 𝑓ₕ₃; maxoutdim=5, pratio=1.0)

# ╔═╡ 923b9503-5a8e-4dab-984d-6ca07d625bb4
Σₕ = cov(fₕ₃, dims=2)

# ╔═╡ 9938d1b8-8f80-45b0-bdff-48cd74e78fa4
P = m.proj

# ╔═╡ 4771733f-83a5-4c13-b0ea-cb54241f7e19
Λ² = diagm(eigvals(Σₕ))

# ╔═╡ d8847644-4d5b-4113-8870-e8c3dba7e1a2
Λ²[Λ².<0] .= eps()

# ╔═╡ 5463ac9d-9bf7-4363-abb9-d1de01a88681
Λ = sqrt(diagm(m.prinvars))

# ╔═╡ 39871429-722a-4745-9a67-62c2961fb268
ffh = P'*𝑓ₕ₃

# ╔═╡ 31a016af-82fa-41a1-b2e8-8adea326c11c
ff = P'*𝑓₃

# ╔═╡ 31cfab8a-e219-484c-9428-1fef8ea16ca5
ff0 = P'*𝑓₀

# ╔═╡ c64a1c15-824a-49d2-ac0f-10573a1472c5
l0 = cov(ff0, dims=2)

# ╔═╡ 8c9e6bd3-c548-4126-a6fb-a100e91f8747
l = cov(ff, dims=2)

# ╔═╡ 24425756-1e29-4e78-9398-6ef1820940b8
lh = Λ^2

# ╔═╡ 4a74c5b1-d19b-4add-a0ca-3a239d8a7c75
l*(I - Diagonal(l0)/Diagonal(l))/(Λ - l0)

# ╔═╡ 3914f4ec-4b70-4f05-9ec1-a9f70223bfd8
inv(Λ)[3, 3]*l[3, 3]*(1-l0[3, 3]/l[3, 3])

# ╔═╡ 94cf8ec3-5c81-4874-a331-aa0f15563808
a = (P*inv(Λ))'*(𝛴²ₕ₃)*(P*inv(Λ))

# ╔═╡ ffb3c3bb-827e-4173-90af-231bc45fef11
𝛴²₃

# ╔═╡ 2a1d988a-21fd-43b1-9701-dcfcf8f3d09b
A = (P*sqrt((I - Diagonal(l0)/Diagonal(l))/(Λ - l0)))'*(𝛴²₃)*P*sqrt((I - Diagonal(l0)/Diagonal(l))/(Λ - l0))

# ╔═╡ f51dc362-e8ab-494b-94f9-7ec63262afe4
B = (P*inv(Λ))'*𝛴²₀*P*inv(Λ)

# ╔═╡ 9ae7ea6f-6f70-4d92-9009-4ed3ccde3945
C = Real.(((I - sqrt(Diagonal(l0))/sqrt(Diagonal(l)))/(sqrt(Λ) - sqrt(l0))))'*ff

# ╔═╡ 5a0bd5e0-e3fb-47dc-8be2-fec23ea865a9
cov(C, dims=2)

# ╔═╡ 2df2e495-5170-4db3-904b-0c272f69b182
D = Real.(sqrt((I - Diagonal(l0)/Diagonal(l))/(Λ - l0)))'*ff0

# ╔═╡ 53142c10-19e9-42ba-b06d-85e326c8c575
cov(D, dims=2)

# ╔═╡ 008b10fc-8e80-4783-ae37-1f18147d6f55
scatter(C[2, :], .-C[3, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Standardised PCA estimate", left_margin=5Plots.mm)

# ╔═╡ c43d7521-0e3d-403c-83b6-d9cce7675931
scatter!(.-𝛽, .-0.1.*𝛼, markersize=1, markercolor=:gray)

# ╔═╡ Cell order:
# ╠═09bee220-1249-11ec-3e2a-4fdde9976a59
# ╟─1cbf17f7-8322-4e3c-a895-163a48332205
# ╟─2f0cb4e5-4424-42d9-8eb7-e8bf128575ad
# ╟─0f10dc15-9153-4d8c-a5e5-c0318c459fa1
# ╟─230c66e5-7898-48af-9544-439677bee137
# ╠═fee5a67f-1678-45fc-83b3-3ffc187c19b3
# ╠═c3e707a4-1e35-4fe0-8e6d-88d0eca27fde
# ╠═5146ae95-d551-49ec-888a-71fc7bf65d14
# ╠═b646056a-7c34-4cb3-aac9-d255646f6b75
# ╟─2fbff3c4-9377-4624-90d6-d4ea7c8da64e
# ╟─404bb5c9-74ab-43a9-8791-67ac23b0c0f8
# ╟─4f01861c-dd52-462f-b340-ceecd3385c8e
# ╠═92670bd1-ec33-4590-9a53-ceca6f25ac0f
# ╠═8629c505-a1e4-4c73-b2d4-44ecc858373d
# ╟─c3c3d6c3-49a0-40a6-b41b-dbdfa20a7d6b
# ╠═9247ba48-f242-4479-a92d-e994b1610279
# ╠═c973721b-c20b-4573-a8fa-a9a9282fc87d
# ╟─4032a036-842b-462b-b75d-ebaaaf4aded9
# ╠═6a957b7d-8fff-46c5-b441-01f8b9d45eb7
# ╟─5686638b-ba3f-4691-9291-d91dfa510aa7
# ╠═76e702f7-521c-4445-aa6c-de28215d8d44
# ╠═b17a0c26-ce25-4e11-84dd-0d4b86371ae5
# ╠═69e0799e-18f4-4f19-9bae-44202c2bba5d
# ╠═cf454ccf-e505-4df9-a3cf-4afba2211d4a
# ╠═8013d3a2-a011-456b-9bfe-3ec781c31aba
# ╟─ce024bfd-b892-4a31-a0d1-1d4befc7b80e
# ╠═542aec21-e623-4e53-ae4b-a7f240bfcb08
# ╠═1daa139a-23c0-4995-bf73-abfbb7c518dd
# ╠═705068d7-033f-49d2-bb59-4b40aa514642
# ╠═f1a685e5-ab99-4a61-a2d9-41ebf33372b0
# ╠═3a7af0db-8e3d-4100-a7c2-dcd37253a9ce
# ╠═7a28c225-bab8-40ee-906e-0c5349accc84
# ╟─4d619965-6eb7-4528-804c-00bbf1572e60
# ╠═7ddd5530-7590-42a7-8e66-54432f717eec
# ╠═a95a3d4c-cb8b-4840-bc67-57fb2be00065
# ╠═688dc093-c173-4eb4-ad12-4ff2e899b9f9
# ╠═f4081452-9314-47a4-948d-2813bda53507
# ╠═5b853331-88c7-4b7f-b26e-8e0f963a5ed5
# ╠═1ee56732-671e-447e-adb8-cce45902b935
# ╠═70d2fe5f-7f7d-4891-8490-5c508c21c6f6
# ╟─bb52a2b3-e856-4785-abb9-78ba15382d11
# ╟─70d2f196-1604-4cf4-8e38-05dd8c099c83
# ╟─009c662a-9940-43ef-b0c9-d55560e6f49b
# ╠═46e7d2a0-8b62-4112-92df-74538f5f16d8
# ╠═595e5022-9099-45ab-b5b0-b92fedb96ba8
# ╟─0d062a65-adca-42c4-8b1c-84095d5335c5
# ╠═0f6b9065-55ad-4b1e-bcc3-2e177a0e331d
# ╠═fd506ba8-9a5b-4235-bc43-b730522dce4f
# ╟─58083319-d649-4c2d-b4b2-c2df7c49ee46
# ╠═2a9a4929-8755-4237-a454-fda125d548ca
# ╟─1762e05f-7eec-4411-865a-eba47531a49c
# ╠═feeaa835-28e0-4a24-905b-53347560ec9a
# ╠═2358f79f-f44d-4e93-b4da-75e7be715882
# ╠═d9fca85a-63b6-4ab5-ab0e-6cd5e68c4e06
# ╠═d700b4ae-8bd5-4437-b515-c21d3da70543
# ╠═06e226a6-0f0f-4ef6-b916-04d51a3789c8
# ╠═ea513127-e9bb-40a7-9ec0-2beee8fa6179
# ╟─f0a7f529-19ac-44f7-8ab1-8d72d596a0ef
# ╠═f946e3d7-15b9-45a9-951b-b3d77fc589f8
# ╠═652b998d-a6b0-4bb7-908e-c3dc7a82764c
# ╠═d192149c-1ef4-4761-9e78-879a97abee57
# ╠═14d5e08c-0808-4352-835e-c173c53fd8d7
# ╠═772d260a-72b3-4c9e-8e65-6368508cfb52
# ╠═0f83a03c-890d-4771-b83f-16d8d5db81e9
# ╠═b9a8e7e4-5f9b-462d-845e-b873f50bfc90
# ╠═f1df27ea-0537-4f4b-afe7-c0692ca7b26d
# ╟─2f987bde-a87a-43a8-8653-0c5812c56a70
# ╟─1d18e8c7-7eff-4b75-ad67-e61bc22e12ea
# ╠═3821aa68-e3b3-4764-a7ff-0b6b0e6bdc3e
# ╠═9d0c3f2b-8cf4-41f9-8fe8-5cd9a19dbd88
# ╠═b0d86eb8-6f94-424a-adb2-eae8e40be1fd
# ╟─e507beed-984e-4f8b-a59b-bb3dcff1416a
# ╠═ea23a142-1c36-4430-a598-2fcf0f2e3b66
# ╠═fe0902aa-b55e-4b72-a35e-77db673747c9
# ╟─9b92548e-ae54-48de-a5c9-4be88ae11228
# ╠═33167762-e2c6-4aba-98b7-58e28ee35ccb
# ╠═35dcd411-a342-4ee2-9146-f45ad39d6341
# ╟─d2404839-53bf-49d3-a9bf-a0ea81471c43
# ╠═e64a04de-9257-4bbb-84c4-007c532d9c72
# ╟─710d51ea-8196-4a7f-91fe-1a9bd3f06d01
# ╠═a4c8677f-6bdc-4bfe-adcc-3130f45b7aaa
# ╠═5ad01546-7daa-441f-bf2c-60e2a6e6f0c7
# ╠═36039419-6df3-45f2-ae2c-77cf6e3b405d
# ╠═d046eb57-d4cb-4eda-9a81-00a234af1d57
# ╠═4e9b0004-da7a-4cb3-afe9-c3365d37471a
# ╠═5e2281e9-360b-41dd-9f4f-38e14d8aea86
# ╠═fa43e761-0b22-4903-aa6c-47050d8c27c0
# ╟─bcd85e11-8bf0-4968-a528-8b56ce14db5b
# ╟─3fe79e11-947e-46e6-a5ef-1bf1b537c857
# ╠═f771887a-77f0-4a4d-9e4c-7e988201197f
# ╠═b636df98-f935-46b4-bb08-be96cee18884
# ╠═c43d6053-4eeb-450a-bca4-bd5d58b179c1
# ╠═fd00b9e4-f2fc-4571-8596-b1e53f8d7286
# ╠═9d13988d-e625-42b5-8e89-86d519027912
# ╠═cb77a71b-7cba-489d-a51b-690cdd281a4c
# ╠═e26f0ae5-4b0b-4886-a0ad-e109dbf9e321
# ╠═dc9c5d21-87fc-404b-b8b7-1a704d890591
# ╠═3eda9fc1-0d54-4cde-94ba-5e611a4773b6
# ╠═0ae23619-0ce3-4e32-9a11-f9a444235c70
# ╠═f64c9529-1f71-4daa-8f4e-6af000288b38
# ╠═c3067e6d-dfaf-4938-8bbf-4b51514fd286
# ╠═a2f4c245-031b-4f1c-9f44-b3947cbc9795
# ╠═ad7aac3b-303a-4ca6-a44f-f11cba8d115f
# ╠═dbdcd6c4-102d-41cc-b218-100f74355b85
# ╠═9d6334c1-7107-4fbb-b383-9a37fc56de74
# ╠═07e9a417-5adb-4182-a98d-fd618feaf675
# ╠═2826e49a-5566-4aa7-be9a-73799dfbebc7
# ╠═c79d0914-d0db-4f5c-b1f2-a6154d063e78
# ╠═eaae76e1-9326-432a-8fc5-b03c3d93e75a
# ╠═b7a9e831-6ab4-4afd-a5da-9d3b5414245b
# ╠═70109d54-cfcd-46c5-8c67-a3b96286ee09
# ╠═9970c6e2-a2c9-442e-8fca-17b8a5c360b9
# ╠═7408597f-8bc6-493a-80f6-79a5201d625b
# ╠═a031cdd3-7e92-45d8-8037-e6695cd588c2
# ╠═816e841b-3f3b-4b8c-8c4b-4cd93c685ab7
# ╠═923b9503-5a8e-4dab-984d-6ca07d625bb4
# ╠═9938d1b8-8f80-45b0-bdff-48cd74e78fa4
# ╠═4771733f-83a5-4c13-b0ea-cb54241f7e19
# ╠═d8847644-4d5b-4113-8870-e8c3dba7e1a2
# ╠═5463ac9d-9bf7-4363-abb9-d1de01a88681
# ╠═39871429-722a-4745-9a67-62c2961fb268
# ╠═31a016af-82fa-41a1-b2e8-8adea326c11c
# ╠═31cfab8a-e219-484c-9428-1fef8ea16ca5
# ╠═c64a1c15-824a-49d2-ac0f-10573a1472c5
# ╠═8c9e6bd3-c548-4126-a6fb-a100e91f8747
# ╠═24425756-1e29-4e78-9398-6ef1820940b8
# ╠═4a74c5b1-d19b-4add-a0ca-3a239d8a7c75
# ╠═3914f4ec-4b70-4f05-9ec1-a9f70223bfd8
# ╠═94cf8ec3-5c81-4874-a331-aa0f15563808
# ╠═ffb3c3bb-827e-4173-90af-231bc45fef11
# ╠═2a1d988a-21fd-43b1-9701-dcfcf8f3d09b
# ╠═f51dc362-e8ab-494b-94f9-7ec63262afe4
# ╠═9ae7ea6f-6f70-4d92-9009-4ed3ccde3945
# ╠═5a0bd5e0-e3fb-47dc-8be2-fec23ea865a9
# ╠═2df2e495-5170-4db3-904b-0c272f69b182
# ╠═53142c10-19e9-42ba-b06d-85e326c8c575
# ╠═008b10fc-8e80-4783-ae37-1f18147d6f55
# ╠═c43d7521-0e3d-403c-83b6-d9cce7675931
