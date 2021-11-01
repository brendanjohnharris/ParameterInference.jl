begin
	import Pkg
    Pkg.activate(mktempdir())
	Pkg.add(url="https://github.com/brendanjohnharris/Catch22.jl#main")
	using Catch22
	Pkg.add(url="https://github.com/tk3369/YeoJohnsonTrans.jl")
	Pkg.develop(path="C:\\Users\\Brendan\\OneDrive - The University of Sydney (Students)\\Honours\\Code\\ParameterInference.jl")
	Pkg.add(path="C:\\Users\\Brendan\\OneDrive - The University of Sydney (Students)\\Honours\\Code\\ParameterInference.jl")
	using ParameterInference, Plots, StatsBase, Statistics, LinearAlgebra, ParameterInference.NonstationaryProcesses, MultivariateStats, StatsPlots
	gr(); fourseas!(); # Plots initialisation
	nothing
end

nₛ = 1000;
𝜂 = 0.25;
𝛼, 𝛽 = randn(nₛ, 2) |> eachcol;


# ? Case 1
𝑓ₕ₁ = hcat(𝛼, 𝛼, 𝛼, 𝛼, 𝛽)';
𝑓₁ = hcat(𝜂.*[𝛼, 𝛼, 𝛼, 𝛼]..., 𝛽)';
𝛴² = cov(𝑓₁');

𝑓̂₁ = 𝑓₁./std(𝑓₁, dims=2);
𝛴̂² = cov(𝑓̂₁');
𝑃̂ =  fit(MultivariateStats.PCA, 𝑓̂₁).proj; #eigvecs(𝛴̂²);
𝑓̂′₁ = 𝑃̂'*𝑓₁;
cov(𝑓̂′₁, dims=2)
sgn1 = sign(cor(𝑓̂′₁[1, :], 𝛼))
sgn2 = sign(cor(𝑓̂′₁[2, :], 𝛽))


𝛴²ₕ = cov(𝑓ₕ₁');
m1 = fit(MultivariateStats.PCA, 𝑓ₕ₁; maxoutdim=5, pratio=1.0)
𝑃ₕ = m1.proj; # eigvecs(𝛴²ₕ)
𝛬ₕ = sqrt(diagm(m1.prinvars)); # Diagonal(sqrt.(abs.(eigvals(𝛴²ₕ))));
𝑓′′₁ = (𝑃ₕ*inv(𝛬ₕ))'*𝑓₁;
sgn3 = sign(cor(𝑓′′₁[1, :], 𝛼))
sgn4 = sign(cor(𝑓′′₁[2, :], 𝛽))
cov(𝑓′′₁, dims=2)

p = scatter(sgn2.*𝑓̂′₁[2, :], sgn1.*𝑓̂′₁[1, :], ylabel="𝛼", xlabel="𝛽", color=:crimson, markersize=2, label=nothing)
# k = kde((𝑓̂′₁[2, :], 𝑓̂′₁[1, :]); bandwidth=(0.5, 0.5))
# plot!(k, levels=3, color=:crimson)
scatter!(sgn4.*𝑓′′₁[2, :], sgn3.*𝑓′′₁[1, :],  color=ParameterInference.cucumber, markersize=2, label=nothing)
p = scatter!(p, 𝛽, 𝜂.*𝛼,  markersize=2, markercolor=:cornflowerblue, label=nothing)
# k = kde((𝛽, 𝜂.*𝛼); bandwidth=(0.5, 0.5))
# plot!(k, levels=3, color=:cornflowerblue)
covellipse!(p, zeros(2), cov(hcat(𝑓̂′₁[2, :], 𝑓̂′₁[1, :])), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=:crimson, linealpha=0.8, label="\$\\textrm{Standardised\\ PCA}: \\rho_{\\alpha} = $(abs(round(cor(𝑓̂′₁[1, :], 𝛼), digits=2))), \\rho_\\beta = $(abs(round(cor(𝑓̂′₁[2, :], 𝛽), digits=2)))\$")
covellipse!(p, zeros(2), cov(hcat(sgn4.*𝑓′′₁[2, :], sgn3.*𝑓′′₁[1, :])), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=ParameterInference.cucumber, linealpha=0.8, label="\$\\textrm{Baseline\\ PCA}: \\rho_{\\alpha} = $(abs(round(cor(𝑓′′₁[1, :], 𝛼), digits=2))), \\rho_\\beta = $(abs(round(cor(𝑓′′₁[2, :], 𝛽), digits=2)))\$")
covellipse!(p, zeros(2), cov(hcat(𝛽, 𝜂.*𝛼)), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=:cornflowerblue, linealpha=0.8, size=(500, 400), xlims=(-3.6, 3.6), ylims=(-2.4, 2.4), label="True parameters", legend=true, foreground_color_legend=nothing, background_color_legend=nothing)
savefig("../figures/$(rand(UInt16)).svg")

# ? Case 2

# ╔═╡ 46e7d2a0-8b62-4112-92df-74538f5f16d8
𝑓ₕ₂ = hcat(𝛼+𝛽/4, 𝛼+𝛽/4, 𝛼+𝛽/4, 𝛼+𝛽/4, 𝛽)';
𝑓₂ = hcat((𝜂.*[𝛼, 𝛼, 𝛼, 𝛼] .+ [𝛽, 𝛽, 𝛽, 𝛽]./4)..., 𝛽)';

# ╔═╡ feeaa835-28e0-4a24-905b-53347560ec9a
𝑓̂₂ = 𝑓₂./std(𝑓₂, dims=2);

# ╔═╡ 2358f79f-f44d-4e93-b4da-75e7be715882
𝛴̂²₂ = cov(𝑓̂₂')

# ╔═╡ d9fca85a-63b6-4ab5-ab0e-6cd5e68c4e06
𝑃̂₂ = fit(MultivariateStats.PCA, 𝑓̂₂).proj; # eigvecs(𝛴̂²₂);

# ╔═╡ d700b4ae-8bd5-4437-b515-c21d3da70543
𝑓̂′₂ = 𝑃̂₂'*𝑓̂₂;
cov(𝑓̂′₂, dims=2)

sgn2 = sign(cor(𝑓̂′₂[2, :], 𝛼))
sgn1 = sign(cor(𝑓̂′₂[1, :], 𝛽))
std(𝑓̂′₂[1, :])./std(𝑓̂′₂[2, :])
std(𝛽)./std(𝜂.*𝛼 .+ 𝛽./4)
# ╔═╡ 06e226a6-0f0f-4ef6-b916-04d51a3789c8
scatter(sgn1.*𝑓̂′₂[1, :], sgn2.*𝑓̂′₂[2, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Standardised PCA estimate")

# ╔═╡ ea513127-e9bb-40a7-9ec0-2beee8fa6179
scatter!(𝛽, 𝜂.*𝛼 .+ 𝛽./4, markersize=2, markercolor=:gray)

# ╔═╡ f946e3d7-15b9-45a9-951b-b3d77fc589f8
𝛴²ₕ₂ = cov(𝑓ₕ₂');

# ╔═╡ 652b998d-a6b0-4bb7-908e-c3dc7a82764c
m2 = fit(MultivariateStats.PCA, 𝑓ₕ₂);

# ╔═╡ d192149c-1ef4-4761-9e78-879a97abee57
𝑃ₕ₂ = m2.proj; #eigvecs(𝛴²ₕ₂);

𝛬ₕ₂ = sqrt(diagm(m2.prinvars)); # Diagonal(sqrt.(abs.(eigvals(𝛴²ₕ₂))))

𝑓′′₂ = (𝑃ₕ₂*inv(𝛬ₕ₂))'*𝑓₂;

# Now pca againt o get underlying params:
𝑓′′₂ = fit(MultivariateStats.PCA, 𝑓′′₂).proj'*𝑓′′₂;
sgn3 = sign(cor(𝑓′′₂[1, :], 𝛽))
sgn4 = sign(cor(𝑓′′₂[2, :], 𝛼))
scatter(sgn4.*𝑓′′₂[2, :], sgn3.*𝑓′′₂[1, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′");

scatter!(𝛽, 𝜂.*𝛼 .+ 𝛽./4, markersize=2, markercolor=:gray)


p = scatter(sgn1.*𝑓̂′₂[1, :], sgn2.*𝑓̂′₂[2, :], ylabel="𝛼", xlabel="𝛽", color=:crimson, markersize=2, label=nothing)
# k = kde((𝑓̂′₁[2, :], 𝑓̂′₁[1, :]); bandwidth=(0.5, 0.5))
# plot!(k, levels=3, color=:crimson)
scatter!(sgn3.*𝑓′′₂[1, :], sgn4.*𝑓′′₂[2, :], color=ParameterInference.cucumber, markersize=2, label=nothing)
p = scatter!(p, 𝛽, 𝜂.*𝛼, markersize=2, markercolor=:cornflowerblue, label=nothing)
# k = kde((𝛽, 𝜂.*𝛼); bandwidth=(0.5, 0.5))
# plot!(k, levels=3, color=:cornflowerblue)
covellipse!(p, zeros(2), cov(hcat(sgn1.*𝑓̂′₂[1, :], sgn2.*𝑓̂′₂[2, :])), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=:crimson, linealpha=0.8, label="\$\\textrm{Standardised\\ PCA}: \\rho_{\\alpha} = $(abs(round(cor(𝑓̂′₂[2, :], 𝛼), digits=2))), \\rho_\\beta = $(abs(round(cor(𝑓̂′₂[1, :], 𝛽), digits=2)))\$")
covellipse!(p, zeros(2), cov(hcat(sgn3.*𝑓′′₂[1, :], sgn4.*𝑓′′₂[2, :])), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=ParameterInference.cucumber, linealpha=0.8, label="\$\\textrm{Baseline\\ PCA}: \\rho_{\\alpha} = $(round(cor(𝑓′′₂[2, :], 𝛼), digits=2)), \\rho_\\beta = $(round(cor(𝑓′′₂[1, :], 𝛽), digits=2))\$")
covellipse!(p, zeros(2), cov(hcat(𝛽, 𝜂.*𝛼)), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=:cornflowerblue, linealpha=0.8, size=(500, 400), xlims=(-6, 6), ylims=(-2.4, 2.4), label="True parameters", legend=true, foreground_color_legend=nothing, background_color_legend=nothing)
savefig("../figures/$(rand(UInt16)).svg")



# ? Case 3
𝜂₃ = 0.1;
𝑓ₕ₃ = hcat(repeat([𝛼.+𝜂₃.*randn(nₛ)],3)..., 𝛽.+𝜂₃.*randn(nₛ), (𝜂₃).*randn(nₛ))';
𝑓₀ = hcat(repeat([𝜂₃.*randn(nₛ)], 3)..., 𝜂₃.*randn(nₛ), 𝜂₃.*randn(nₛ))';
𝑓₃ = hcat(repeat([𝜂.*𝛼 .+ 𝜂₃.*randn(nₛ)], 3)..., 𝛽.+𝜂₃.*randn(nₛ), 𝜂₃.*randn(nₛ))';


# * Standardised PCA


# ╔═╡ a4c8677f-6bdc-4bfe-adcc-3130f45b7aaa
𝑓̂₃ = 𝑓₃./std(𝑓₃, dims=2);

# ╔═╡ 5ad01546-7daa-441f-bf2c-60e2a6e6f0c7
𝛴̂²₃ = cov(𝑓̂₃');

# ╔═╡ 36039419-6df3-45f2-ae2c-77cf6e3b405d
𝑃̂₃ = fit(MultivariateStats.PCA, 𝑓̂₃).proj;# eigvecs(𝛴̂²₃);

# ╔═╡ d046eb57-d4cb-4eda-9a81-00a234af1d57
𝑓̂′₃ = 𝑃̂₃'*𝑓̂₃;
sgn2 = sign(cor(𝑓̂′₃[2, :], 𝛽))
sgn1 = sign(cor(𝑓̂′₃[1, :], 𝛼))

cov(𝑓̂′₃, dims=2) # So actually 3 dim

# ╔═╡ 4e9b0004-da7a-4cb3-afe9-c3365d37471a
scatter(𝑓̂′₃[1, :], 𝑓̂′₃[2, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Standardised PCA estimate")
cor(𝑓̂′₃[1, :], 𝛼)
std(𝑓̂′₃[1, :])./std(𝑓̂′₃[2, :])
std(𝛽)./std(𝜂.*𝛼)
# ╔═╡ 5e2281e9-360b-41dd-9f4f-38e14d8aea86
scatter!(𝛽, 𝜂.*𝛼, markersize=2, markercolor=:gray)

# ╔═╡ bcd85e11-8bf0-4968-a528-8b56ce14db5b
# * Baseline PCA

# ╔═╡ f771887a-77f0-4a4d-9e4c-7e988201197f
𝛴²ₕ₃ = cov(𝑓ₕ₃');

# ╔═╡ b636df98-f935-46b4-bb08-be96cee18884
𝛴²₀ = cov(𝑓₀');

# ╔═╡ c43d6053-4eeb-450a-bca4-bd5d58b179c1
𝛴²₃ = cov(𝑓₃');

# ╔═╡ beafaf38-3753-4e9f-bf6d-69b98439fe18
m3 = fit(MultivariateStats.PCA, 𝑓ₕ₃)

# ╔═╡ fd00b9e4-f2fc-4571-8596-b1e53f8d7286
𝑃ₕ₃ = m3.proj;#eigvecs(𝛴²ₕ₃);

# ╔═╡ 9d13988d-e625-42b5-8e89-86d519027912
𝛬ₕ₃ = sqrt(diagm(m3.prinvars)); #Diagonal(sqrt.(abs.(eigvals(𝛴²ₕ₃))))

# ╔═╡ 4f5c167d-dbde-4fed-bd2d-63b0100e2d1a
𝛴²ₕ′ = diagm(m3.prinvars)

# ╔═╡ 82841686-b078-4ba2-b5ce-d748070437d2
𝛴²₀′ = cov(𝑃ₕ₃'*𝑓₀, dims=2)

# ╔═╡ 8dded5c7-f4ea-4ee5-b0b9-59b21c7f3097
𝛴²′ = cov(𝑃ₕ₃'*𝑓₃, dims=2)

# ╔═╡ cb77a71b-7cba-489d-a51b-690cdd281a4c
𝑉 = I - inv(sqrt(Diagonal(𝛴²′)))*sqrt(Diagonal(𝛴²₀′));

# ╔═╡ b00f965a-18a1-4ccf-85a9-ae79d588cab6
𝑆 = 𝑉/(𝛬ₕ₃ - sqrt(Diagonal(𝛴²₀′)))

# ╔═╡ 0ae23619-0ce3-4e32-9a11-f9a444235c70
𝑓′′₃ = 𝑆'*𝑃ₕ₃'*𝑓₃;
# Now pca againt o get underlying params:
𝑓′′₃ = fit(MultivariateStats.PCA, 𝑓′′₃).proj'*𝑓′′₃;
cov(𝑓′′₃, dims=2)
sgn3 = sign(cor(𝑓′′₃[1, :], 𝛽))
sgn4 = sign(cor(𝑓′′₃[2, :], 𝛼))

# ╔═╡ ddb9528f-b57c-4333-9347-8a56aa6890c4
#f′′₃ = (𝑓′′₃./std(𝑓′′₃, dims=2))*((sqrt(Diagonal(𝛴²′)) - sqrt(Diagonal(𝛴²₀′)))/())

# ╔═╡ c3067e6d-dfaf-4938-8bbf-4b51514fd286
𝑓′′₀ = (𝑃ₕ₃*inv(𝛬ₕ₃)*𝑉)'*𝛴²₀*(𝑃ₕ₃*inv(𝛬ₕ₃)*𝑉); # Not 0 because the transform is a strong approximation

# ╔═╡ a2f4c245-031b-4f1c-9f44-b3947cbc9795
cov(𝑓′′₃, dims=2)

# ╔═╡ ad7aac3b-303a-4ca6-a44f-f11cba8d115f
𝑓′′₀

# ╔═╡ dbdcd6c4-102d-41cc-b218-100f74355b85
scatter(𝑓′′₃[2, :], .-𝑓′′₃[1, :], aspect_ratio=:equal, ylabel="𝛼′", xlabel="𝛽′", title="Baseline PCA estimate", left_margin=5Plots.mm);

# ╔═╡ 9d6334c1-7107-4fbb-b383-9a37fc56de74
scatter!(.-𝛽, .-𝜂.*𝛼, markersize=1, markercolor=:gray)

# ╔═╡ 07e9a417-5adb-4182-a98d-fd618feaf675
scatter(𝛽, 𝑓′′₃[2, :]) # Compairwise to true parameters

# ╔═╡ 008512c1-3a44-42b4-8695-2a9f4d88ef01
scatter(𝛼, 𝑓′′₃[1, :]) # Compairwise to true parameters




p = scatter(sgn2.*𝑓̂′₃[2, :], sgn1.*𝑓̂′₃[1, :], ylabel="𝛼", xlabel="𝛽", color=:crimson, markersize=2, label=nothing)
# k = kde((𝑓̂′₁[2, :], 𝑓̂′₁[1, :]); bandwidth=(0.5, 0.5))
# plot!(k, levels=3, color=:crimson)
scatter!(sgn3.*𝑓′′₃[1, :], sgn4.*𝑓′′₃[2, :], color=ParameterInference.cucumber, markersize=2, label=nothing)
p = scatter!(p, 𝛽, 𝜂.*𝛼, markersize=2, markercolor=:cornflowerblue, label=nothing)
# k = kde((𝛽, 𝜂.*𝛼); bandwidth=(0.5, 0.5))
# plot!(k, levels=3, color=:cornflowerblue)
covellipse!(p, zeros(2), cov(hcat(sgn2.*𝑓̂′₃[2, :], sgn1.*𝑓̂′₃[1, :])), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=:crimson, linealpha=0.8, label="\$\\textrm{Standardised\\ PCA}: \\rho_{\\alpha} = $(abs(round(cor(𝑓̂′₃[1, :], 𝛼), digits=2))), \\rho_\\beta = $(abs(round(cor(𝑓̂′₃[2, :], 𝛽), digits=2)))\$")
covellipse!(p, zeros(2), cov(hcat(sgn3.*𝑓′′₃[1, :], sgn4.*𝑓′′₃[2, :])), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=ParameterInference.cucumber, linealpha=0.8, label="\$\\textrm{Baseline\\ PCA}: \\rho_{\\alpha} = $(round(cor(𝑓′′₃[2, :], 𝛼), digits=2)), \\rho_\\beta = $(round(cor(𝑓′′₃[1, :], 𝛽), digits=2))\$")
covellipse!(p, zeros(2), cov(hcat(𝛽, 𝜂.*𝛼)), n_std=2.25, fillcolor=XYZA(1, 1, 1, 0.0), linecolor=:cornflowerblue, linealpha=0.8, size=(500, 400), xlims=(-3.6, 3.6), ylims=(-4.8, 4.8), label="True parameters", legend=true, foreground_color_legend=nothing, background_color_legend=nothing)
savefig("../figures/$(rand(UInt16)).svg")
