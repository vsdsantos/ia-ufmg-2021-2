### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6021fa5e-d35c-4311-a83c-ac33c665ab02
using Plots, IterTools, Combinatorics, Random, Distributions, StatsBase, PlutoUI

# ╔═╡ 0091b053-f24e-406c-9b48-04882356ad86
md"""# Lista 1 - IA - 2021/2
Aluno: Victor Silva dos Santos
"""

# ╔═╡ 3475316d-8524-430d-918e-ece3a5b76bc9
md"Código fonte completo disponível em:"

# ╔═╡ bba34a41-77db-4225-966d-fabd58896074
html"<a href='https://github.com/zuckberj/ia-ufmg-2021-2/blob/main/lista1.jl'>GiHub</a>"

# ╔═╡ d0ceb131-98ea-4a6d-979f-42bd0bbeecb6
md"É possível executar este notebook pelo link:

*o serviço Binder é relativamente demorado e pode necessitar um restart do notebook e execussão da primeira célula para funcionar completamente.
"

# ╔═╡ 53e4ca4f-4ead-40fd-b6c1-6bdc63352a6b
html"<a href='
https://binder.plutojl.org/v0.17.1/open?url=https%253A%252F%252Fraw.githubusercontent.com%252Fzuckberj%252Fia-ufmg-2021-2%252Fmain%252Flista1.jl'>Binder</a>"

# ╔═╡ 1a39b939-10f9-4ad3-ac30-3bc2d6934071
md"## Problema 1 -- N-Queens"

# ╔═╡ 07054dd1-386e-4deb-8ff8-3ef912d43d98
md"Função do problema"

# ╔═╡ c1ba8e98-6a4e-45e6-8fcb-4cde49be7fac
function queen_fit(X::Vector{<:Integer})
	D = length(X)
	pos = collect(1:D)
	fit = 0
	for i in pos
		P = X[i]
		for j in (pos .- i)
			if j == 0
				continue
			end
			if P+j == X[i+j] || P-j == X[i+j]
				fit += 1
			end
		end
	end
	fit
end

# ╔═╡ 35af2c60-74d7-4719-a6ca-9124e6e95367
md"definição das estratégias do AG"

# ╔═╡ f59946ea-c0d6-404c-b453-21e350d9e039
D_nq = 8

# ╔═╡ 88af6f31-2303-4fa8-b5b3-4d36fb9492ab
md"População inicial"

# ╔═╡ bcf5aa9e-30a1-40af-b1ed-2b18c6534e6b
md"roda algoritmo"

# ╔═╡ 6fbc7788-4736-4346-a08b-a0e0e99f363e
md"## Problema 2 -- Funções teste
### Função esfera"

# ╔═╡ 6fe88ef4-0342-4632-bb98-e3e36e2181e4
md"### Função Rastringin"

# ╔═╡ cf8b1a3f-70d5-490f-83bb-881fe73c0c16
D_11 = 10

# ╔═╡ 0360fb13-f186-40a4-9ee6-cf7fb80dd659
md" ## Algoritmo Genético"

# ╔═╡ 4dab05b9-b1b2-453e-8f2f-09c1632b3d48
# Configurações do algoritmo
struct GAStrategy
    selection # tipo de seleção
	generation_gap::Real # λ/μ
    crossover # tipo de crossover
    crossover_prob::Real # prob de crossover
    mutation
    mutation_σ::Real
    mutation_prob::Real
    error_tolerance::Real
	strike_tol::Integer
    max_iter::Integer
end

# ╔═╡ e66bff7f-78a1-4317-b7f4-e8287d7a0875
strat_nq = GAStrategy(:ranking, 0.9, # selection
			   :default, 0.8, # reproduction
			   :default, 1, 0.1, # mutation
				0, 10, 75)

# ╔═╡ 2c217353-803a-4556-a4dc-1cdff404e7be
strat1 = GAStrategy(:ranking, 1, # selection
				   :default, 1, # reproduction
				   :default, 2, 0.05, # mutation
					1e-6, 1, 1000)

# ╔═╡ c0ad6d28-4046-47cc-9ae6-6012b7f21ce9
strat11 = GAStrategy(:ranking, .9, # selection
				   :default, .9, # reproduction
				   :default, 30, 0.08, # mutation
					1e-10, 5, 1000)

# ╔═╡ e1464a65-80b2-415b-9ab2-5547edb12f74
# Classe do Algoritmo Genérico
struct GA
    f::Function # função a ser minimizada/maximizada
    LB::Array{<:Number} # limites inferiores
    UB::Array{<:Number} # limites superiores
    strategy::GAStrategy # configurações
	objective::Symbol # max ou min
end

# ╔═╡ 8be18e7b-a996-46e7-946f-ceeb82de8bd1
ga_nq = GA(queen_fit, [1], [1], strat_nq, :min)

# ╔═╡ 4546f9d3-1f6e-4a04-9bc6-8eaf44c4f7eb
function selection_prob(ga::GA, fitness::Vector{<:Real})
	N = length(fitness)
	
	s = ga.strategy
	if s.selection == :proportional
		PS = fitness ./ sum(fitness)
	elseif s.selection == :ranking
		s = rand() + 1
    	idx = sortperm(fitness, rev=true)
		rank = collect(N-1:-1:0)
		rank[idx] = rank
		PS = (2-s)/N .+ 2 .* rank .* (s-1) / (N*(N-1))
	elseif s.selection == :ranking_exp
		idx = sortperm(fitness, rev=true)
		rank = collect(N-1:-1:0)
		rank[idx] = rank
		PS = 1 .- exp.(-rank)
		PS ./= sum(PS)
	else
		PS = (1/N).*ones(N)
	end
	PS
end

# ╔═╡ c9300ca8-205c-44ce-a74d-bc1af03a8a48
function roulette_selection(X::Vector{<:Any}, PS::Vector{<:Real}, λ::Integer)
	[sample(X, pweights(PS)) for i in 1:λ]
end

# ╔═╡ 2a85f4f2-91c8-4e58-a06c-c80cb4b0d7fe
function sus_selection(X::Vector{<:Any}, PS::Vector{<:Real}, λ::Integer)
	r0 = rand()/λ
	r = r0 .+ collect(0:λ-1)./λ
	
	Xr = Vector{typeof(X[1])}(undef, λ)
	a = cumsum(PS)
	for i in 1:λ
		j = 1
		while a[j] <= r[i]
			j += 1
		end
		Xr[i] = X[j]
	end
	Xr
end

# ╔═╡ e1cc07c2-d7d0-4344-8f32-a8b49a357e4b
# Seleção dos indivíduos
function selection(ga::GA, X::Vector{<:Any}, fitness::Vector{<:Real}, age=0)
    N = length(X)
    s = ga.strategy
	
    PS = selection_prob(ga, fitness)
    
    if s.selection == :reduce_by_age
        λ = Integer(ceil(N * (s.generation_gap*(1-age))))
        Xr = sus_selection(X, PS, λ)
    else
		λ = Integer(ceil(N * (s.generation_gap)))
        Xr = sus_selection(X, PS, λ)
    end
    
    Xr
end

# ╔═╡ 63364c03-04db-414b-a58b-c057da38166e
# Cria uma geração aleatória de N indivíduos com D variáveis
function rand_X(ga::GA, D::Integer, N::Integer, T=Float64)
    X = Vector{Vector{T}}(undef, N)
    for i in 1:N
        if T <: AbstractFloat
            v = ga.LB .+ rand(T, D).*(ga.UB.-ga.LB)
		elseif T <: Bool
            v = trunc.(T, ga.LB .+ rand(D).*(ga.UB.-ga.LB))
		else
        end
        X[i] = v
    end
    X
end

# ╔═╡ 4b3b752b-54c1-44ff-baba-232a0a57ff08
evaluate_f(ga::GA, X::Vector{<:Any}) = ga.f.(X)

# ╔═╡ 53044516-2a6f-433c-b2aa-5855a02009c1
md"### Representação por Permutação"

# ╔═╡ d25316bb-a1cb-49e6-bf1b-0bfd7b678791
function rand_X_perm(ga::GA, N::Integer, D::Integer)
	X = Vector{Vector{Integer}}(undef, N)
    for i in 1:N
		v = collect(1:D)
		shuffle!(v)
		X[i] = v
	end
	X
end

# ╔═╡ f8c79585-33aa-4627-bf2d-8deebd9ca779
X0_nq = rand_X_perm(ga_nq, 20, D_nq)

# ╔═╡ c1692fd0-7154-4757-8e78-01d99795a0e4
function cyclic_grouping(XA::Vector{<:Integer}, XB::Vector{<:Integer})
	D = length(XA)
	arr = []
	
	pos = Set(1:D)
	
	while length(pos) > 0
		
		P0 = minimum(pos)
		
		res = Set([])
		push!(res, P0)
		P = indexin(XA[P0], XB)[1]
		while P != P0
			push!(res, P)
			P = indexin(XA[P], XB)[1]
		end
		
		push!(arr, res)

		pos = setdiff(Set(1:D), union(arr...))
	end
	arr
end

# ╔═╡ 7d4845c3-2043-44cd-83c6-bcebf0a01ea2
# bitflip
function reproduction(ga::GA, XA::Vector{<:Integer}, XB::Vector{<:Integer})

	arr = cyclic_grouping(XA, XB)
	
	D = length(XA)
	childA, childB = Vector{Integer}(undef, D), Vector{Integer}(undef, D)

	if rand() < ga.strategy.crossover_prob
		ord = rand() < 0.5
		for s in arr
			s = [s...]
			if ord
				childA[s], childB[s] = XB[s], XA[s]
			else
				childA[s], childB[s] = XA[s], XB[s]
			end
			ord = ! ord
		end
	else
		childA, childB = XA, XB	
	end
	
    childA, childB
end

# ╔═╡ 57b6b893-bd08-4b54-ba77-efb1484a768b
function mutate_int!(X::Vector{<:Any}, ga::GA, age=0)
	N = length(X)
    D = length(X[1])
	
	pos = Set(1:D)

	if rand() < ga.strategy.mutation_prob
		for i in 1:N
			idx1 = rand(pos)
			idx2 =	rand(setdiff(pos, Set(idx1)))
			
			if idx2 < idx1
				i = idx1
				idx1 = idx2
				idx2 = i
			end
			
			p = rand()
			if p < 1/4
				# swap
				v1 = X[i][idx1]
				X[i][idx1] = X[i][idx2]
				X[i][idx2] = v1
			elseif p < 2/4
				# insert
				vs = X[i][idx1+1:idx2]
				permute!(vs, [length(vs); collect(1:length(vs)-1)])
				X[i][idx1+1:idx2] = vs
			elseif p < 3/4
				# scramble
				vs = X[i][idx1:idx2]
				shuffle!(vs)
				X[i][idx1:idx2] = vs
			else
				# invert
				vs = X[i][idx1:idx2]
				reverse!(vs)
				X[i][idx1:idx2] = vs
			end
		end
	end
    X
end


# ╔═╡ 707c0054-5bed-4909-bd2c-f2392948ca0b
md"### Representação por Real"

# ╔═╡ 656f7acb-5b15-44d9-b225-074280b597ea
# crossover aritimético total
function reproduction(ga::GA, XA::Vector{<:Real}, XB::Vector{<:Real})
    if rand() < ga.strategy.crossover_prob
        childA, childB = Float64[], Float64[]
        for (a, b) in zip(XA, XB) 
            α = rand()
            push!(childA, a*α + b*(1-α))
            push!(childB, b*α + a*(1-α))
        end
    else
        childA, childB = XA, XB
    end
    childA, childB
end

# ╔═╡ f6c638ba-0248-4b88-8dce-e0c35608a092
md"### Representação por Bit"

# ╔═╡ e6b953f3-1d3d-4a45-a928-6ee8e283b588
md"## Funções de teste"

# ╔═╡ 38318598-22c4-49a2-900e-6d63fc94add0
md"### Função esfera"

# ╔═╡ 8238c94e-0c62-4cd3-bbc5-b33f08c30914
begin
	f1a = -1400
	f1(X) = sum((X).^2) + f1a
end

# ╔═╡ 0ffcf586-9efb-479a-b984-2b89e3292cba
ga1 = GA(f1, [-100,-100], [100, 100], strat1, :min)

# ╔═╡ 5b6f6cf4-92bb-4a8e-847e-8f7ed3a4e53d
X0_1 = rand_X(ga1, 2, 100, Float64)

# ╔═╡ dd586276-6afe-4016-beb0-fe1bc59b7fb5
let
	x = range(-100,100, length=200)
	y = range(-100,100, length=200)
	xy = map(x->collect(x), Iterators.product(x,y))
	z = f1.(xy)
	
	s = surface(x,y,z)
	c = contour(x,y,z, fill=true)
	plot(s, c; layout=(2,1))
end

# ╔═╡ ae6c7f08-6f7d-49e7-9825-1c4d69dea2dd
md"### Rotated High Conditioned Elliptic Function"

# ╔═╡ 977cc41a-a5c3-4c63-97b0-752b79a8b13e
md"### Rastringin function"

# ╔═╡ d7d5e1b9-3082-4e25-a2d3-8d0e61758289
md"## Helping functions"

# ╔═╡ 885f143e-4708-4c37-9cac-0cf99b4f0682
function plot_chess(X::Vector{<:Integer})
	D = length(X)
	M = [i%2 == 0 ? (j%2==0 ? 1. : 0.5) : (j%2==1 ? 1. : 0.5) for i = 1:D, j = 1:D]
	for (i,x) in enumerate(X)
		M[x,i] = 0
	end
	Gray.(M)
end

# ╔═╡ bb8e4d4b-04e1-4566-bf96-4860fa4e2735
begin
@userplot GenPlot2D
@recipe function f(gp2D::GenPlot2D)
    ga, Xs, i, mode = gp2D.args
    
    if i > 1
        Xp = Xs[i-1]
        xp, yp = map(x->x[1], Xp), map(x->x[2], Xp)
    end
    
    Xn = Xs[i]
    xn, yn = map(x->x[1], Xn), map(x->x[2], Xn)

	k = 0
	
    if i > 1 && mode == :follow
        minx, miny = minimum([xn;xp]), minimum([yn;yp])
        maxx, maxy = maximum([xn;xp]), maximum([yn;yp])
        x = range(minx - abs(minx*k), maxx + abs(maxx*k), length=100)
        y = range(miny - abs(miny*k), maxy + abs(maxy*k), length=100)
    else
        x = range(ga.LB[1], ga.UB[1], length=100)
        y = range(ga.LB[2], ga.UB[2], length=100)
    end
    xy = map(x->collect(x), Iterators.product(x,y))
    z = ga.f.(xy)
    
    @series begin
        seriestype := :contour
        fill --> false
        x, y, z
    end
    
    if i > 1
        @series begin
            seriestype := :scatter
            label --> false
            markeralpha --> 0.2
            seriescolor --> :blue
            xp, yp
        end
    end
    
    @series begin
        seriestype := :scatter
        label --> "Gen "*string(i)
        seriescolor --> :green
        xn, yn
    end
end
end

# ╔═╡ d6adfd3c-258e-4cb9-ade9-bf31d0e74b19
let
	p = genplot2d(ga1, [X0_1], 1, :fixed)
	plot!(p, title="Primeira Geração")
end

# ╔═╡ 1b1c87ab-9c57-4c6e-9651-b0fc58a352ca
grayencode(n::Integer) = n ⊻ (n >> 1)

# ╔═╡ 6cf2bcc2-0ca3-4946-adaa-21f6c700ccb6
function graydecode(n::Integer)
    r = n
    while (n >>= 1) != 0
        r ⊻= n
    end
    return r
end

# ╔═╡ 761370a2-7cb3-4142-8845-f1bb3fa3b195
# bitflip
function reproduction(ga::GA, XA::Vector{<:BitArray}, XB::Vector{<:BitArray})
    if rand() < ga.strategy.crossover_prob
        childA, childB = Integer[], Integer[]
        for (a, b) in zip(XA, XB)
            r = rand(typeof(a))
            newgenA = (~r & grayencode(a)) | (r & grayencode(b))
            newgenB = (~r & grayencode(b)) | (r & grayencode(a))
            push!(childA, graydecode(newgenA))
            push!(childB, graydecode(newgenB))
        end
    else
        childA, childB = XA, XB
    end
    childA, childB
end

# ╔═╡ ca3796ef-2c3e-486b-b571-d17a278ad1c9
function reproduce_gen(ga::GA, X::Vector{<:Any}, N::Integer)
	
    comb = collect(combinations(1:length(X), 2))
    
    newX = Vector{typeof(X[1])}(undef, N)
    i = 1
    while i <= N
        comb = shuffle(comb)
        for k in 1:length(comb)
            if i >= N
                break
            end
            a, b = comb[k]
            childA, childB = reproduction(ga, X[a], X[b])
            newX[i] = childA
            newX[i+1] = childB
            i += 2
        end
    end

	newX
end

# ╔═╡ db6af83c-fc12-43f6-9a4b-459fb659d132
function saturate!(ga, X::Array{<:Any})
	D = length(X)
	for i in 1:D
		if X[i] < ga.LB[i]
			X[i] = ga.LB[i]
		elseif X[i] > ga.UB[i]
			X[i] = ga.UB[i]
		end
	end
	X
end

# ╔═╡ c2a65bb4-ff08-4f0b-ade7-a2a2800bf1cc
function mutate_real!(X::Vector{<:Any}, ga::GA, age=0)
    N = length(X)
	D = length(X[1])

	for i in 1:N
	    if rand() < ga.strategy.mutation_prob
			if ga.strategy.mutation == :reduce_by_age
	        	X[i] += rand(Normal(0, ga.strategy.mutation_σ*(1-age)), D)
			else
				X[i] += rand(Normal(0, ga.strategy.mutation_σ), D)
			end
			saturate!(ga, X[i])
	    end
	end
    X
end

# ╔═╡ 2b2509dd-2ed6-4f9f-9a28-a273d44fe5ea
function mutate!(X::Vector{<:Any}, ga::GA, age=0.0)
	T = typeof(X[1])
	if T <: AbstractFloat
		mutate_real!(X, ga, age)
	else
		mutate_int!(X, ga, age)
	end
end

# ╔═╡ a0092fbe-c792-4e92-b5e3-ad79ef77f5be
function bitarr_to_int(arr, val = 0)
    v = 2^(length(arr)-1)
    for i in eachindex(arr)
        val += v*arr[i]
        v >>= 1
    end
    return val
end

# ╔═╡ c18e2efb-8710-4a67-9689-ede1fe877b2d
function mutate!(X::Vector{<:BitArray}, ga::GA, age=0)
    D = length(X)
    T = typeof(X[1])
    
#     if rand() > ga.strategy.mutation_prob
        for i in 1:D
        
            Δ = trunc(T, ga.LB[i] - ga.UB[i])
        
            x = X[i] + Δ
            
            flip_arr = [0; rand(ndigits(typemin(Int8), base=2)-1) .< ga.strategy.mutation_prob]
            flip = bitarr_to_int(flip_arr)
            
            x = graydecode(grayencode(x) ⊻ trunc(T, flip))
            
            X[i] = x - Δ
        end
	saturate!(ga, X)
#     end
    X
end

# ╔═╡ 60fe54f6-837e-4cac-903a-3db308f71d8f
function evolve_gen(ga::GA, X::Vector{<:Any}, fitness::Vector{<:Real}, age=0)
    N = length(X)
    D = length(X[1])
    s = ga.strategy

	# realiza a seleção dos indivíduos para reprodução
    Xred = selection(ga, X, fitness, age)

	# reprodução dos indivíduos
	newX = reproduce_gen(ga, Xred, N)
	
	# mutação dos indivíduos
    mutate!(newX, ga, age)
	
    newX
end

# ╔═╡ c9a7c598-230b-41fa-a992-747c7e640da9
# roda o algoritmo
function run_ga(ga::GA, X0::Vector{<:Any})

	# minimizar ou maximizar a função
    sig = ga.objective == :max ? 1 : -1

	# primeira avaliação de fitness
    fitness0 = evaluate_f(ga, X0) .* sig
    new_Xs = X0
    fitness = fitness0
    
    i = 1

	# histórico
    xhist = [new_Xs]
    fithist = [fitness]

	strike = 0
	
    while i <= ga.strategy.max_iter
		
		# executa os passos de evolução da geração atual
        new_Xs = evolve_gen(ga, new_Xs, fitness, i/ga.strategy.max_iter)
        
        push!(xhist, new_Xs)
        
        fitness0 = fitness

		# avalia a nova geração
        fitness = evaluate_f(ga, new_Xs) .* sig
        
        push!(fithist, fitness)

		# verifica a convergência
        e = abs.(median(fitness) .- median(fitness0))./abs.(median(fitness0))
        
        if e < ga.strategy.error_tolerance
			strike += 1
			if strike >= ga.strategy.strike_tol
            	break
			end
		else
			strike = 0
        end
        
        i += 1
    end
    
    xhist, fithist.*sig
end

# ╔═╡ 5b4588fa-c73b-49b8-a3ec-9b0b30259f40
xs_nq, fits_nq = run_ga(ga_nq, X0_nq)

# ╔═╡ eabccb4b-9890-428c-9d43-dbab84fd08cc
let
	
	md"""
	Número de gerações: $(length(fits_nq))
	
	Quantidade de avaliações da função objetivo: $(length(fits_nq)*length(X0_nq))

	Permutações possíveis: $(factorial(big(D_nq), 0))
	
	Valor mínimo encontrado: $(minimum(fits_nq[end]))
	"""
end

# ╔═╡ f95a1f72-8333-485f-8756-e52187557a78
md"Seleção da geração: $(@bind i Slider(1:length(xs_nq)))"

# ╔═╡ b6790a2a-bc7a-4e16-ab8c-e998d2af5c31
plot(plot_chess(xs_nq[i][argmin(fits_nq[i])]), title="Gen $(i)")

# ╔═╡ 3699b074-1242-4a35-bee2-40668bcfd5d7
begin
	plot(1:length(xs_nq), mean.(fits_nq), ribbon=std.(fits_nq), title="Dispersão do fitness por geração", label="média")
	plot!(1:length(xs_nq), minimum.(fits_nq), label="melhor fit")
end

# ╔═╡ 40e315bf-49fb-4e80-91c2-5ee237c08d0a
xs_1, fits_1 = run_ga(ga1, X0_1)

# ╔═╡ e598a7e2-a059-47a3-bee5-23890fc4994b
md"""
Número de gerações: $(length(fits_1))

Quantidade de avaliações da função objetivo: $(length(fits_1)*length(X0_1))

Valor mínimo encontrado: $(minimum(fits_1[end]))

Valor mínimo da função: $(f1a)
"""

# ╔═╡ 7b4128c7-fff5-4bf0-b673-46a7ebc818dd
let
	p1 = genplot2d(ga1, xs_1, length(fits_1), :fixed)
	plot!(p1, title="Geração final (visão global)")
end

# ╔═╡ cd96b5e2-d4ae-4b25-aecd-efc02ee96f49
let
	p1 = genplot2d(ga1, xs_1, length(fits_1), :follow)
	plot!(p1, title="Geração final (visão local)")
end

# ╔═╡ 75ecda46-8672-4a2d-a051-28132373ab23
begin
	anim1 = @animate for i in 1:length(xs_1)
		genplot2d(ga1, xs_1, i, :fixed)
	end
	md"#### Evolução"
end

# ╔═╡ c305594a-a65d-4c2f-8177-99a334cbebd6
let
	gif(anim1, fps=10)
end

# ╔═╡ fc1541fc-892c-49a4-80e0-3829c3dde0d7
begin
	plot(1:length(xs_1), mean.(fits_1), ribbon=std.(fits_1), title="Dispersão do fitness por geração", label="média")
	plot!(1:length(xs_1), minimum.(fits_1), label="melhor fit")
end

# ╔═╡ 9838e06b-a30f-426f-b319-51bcf54d45d7
function bitarray_to_int(ga::GA, ba::BitArray)
    l = length(ba)
    k = sum(ba .* (2 .^ (l-1:-1:0)))
    ga.LI + (ga.LS - ga.LI)*k/(2^l - 1)
end

# ╔═╡ 03cf431e-3d6c-4167-81bf-9df45ce6182b
function int_to_bitarray(ga::GA, x::Integer)
    parse.(Bool, split(bitstring(x), ""))
end

# ╔═╡ 5dd415b1-ae92-4e59-9fc5-7060dde228ab
function sign(xi)
    if xi < 0
        return -1.
    elseif xi == 0
        return 0.
    else
        return 1.
    end
end     

# ╔═╡ 885400f7-32af-42d6-b7a7-68000228263b

function x_hat(xi)
    if xi == 0
        return 0.
    end
    log(abs(xi))
end
   

# ╔═╡ a9f4ec4e-4590-4114-a920-11b2654e0991
 
function c1f(xi)
    if xi > 0
        return 10.
    end
    5.5
end


# ╔═╡ 57a57096-97a9-4ead-97bb-03cc5dcf6bd7

function c2f(xi)
    if xi > 0
        return 7.9
    end
    3.1
end
    


# ╔═╡ 19d4d957-b1db-4c09-b808-3eee5463ff68

function Tosz(X)
    xh = x_hat.(X)
    c1, c2 = c1f.(X), c2f.(X)
    D = length(X)
    X[1] = sign(X[1]) * exp(xh[1] + 0.049 * (sin(c1[1]*xh[1]) + sin(c2[1]*xh[1])))
    X[D] = sign(X[D]) * exp(xh[D] + 0.049 * (sin(c1[D]*xh[D]) + sin(c2[D]*xh[D])))
    X
end


# ╔═╡ 78f05a6c-c7f9-450a-ae86-3b1777c89dc3

function Tasz(X, β)
    D = length(X)
    for i in 1:D
        if X[i] > 0
            X[i] = X[i]^(1 + β*sqrt(X[i])*(i-1)/(D-1))
        end
    end
    X
end


# ╔═╡ 68738ecd-2769-4b8a-be9b-a138745ca829
function Α(α, D)
    m = zeros(D, D)
    for i in 1:D
        m[i,i] = α^((i-1)/(2*(D-1)))
    end
    m
end

# ╔═╡ a235d168-76cb-4c1e-8d72-80b55a13b97d
begin
	f11a = -400
	
	function f11(X)
	    D = length(X)
	    Z = Α(10, D) * Tasz(Tosz(5.12.*X./100), 0.2)
	    sum(Z.^2 .- 10cos.(2π*Z) .+ 10) .+ f11a
	end
end

# ╔═╡ 0e6096a2-28b5-42a0-9ca6-c01268e1b28f
ga11 = GA(f11, -100 .* ones(D_11), 100 .* ones(D_11), strat11, :min)

# ╔═╡ b13ba3ef-f838-4758-ae04-59e8be85e250
X0_11 = rand_X(ga11, D_11, 200, Float64)

# ╔═╡ f162da3d-0165-4417-9f88-0ceb31869f88
xs_11, fits_11 = run_ga(ga11, X0_11)

# ╔═╡ 23a16641-13e5-4a35-895e-8752c9d4e0dd
begin
	plot(1:length(xs_11), mean.(fits_11), ribbon=std.(fits_11), title="Dispersão do fitness por geração", label="média")
	plot!(1:length(xs_11), minimum.(fits_11), label="melhor fit")
end

# ╔═╡ d9dec120-048b-49fc-9684-ce28c69a56e1
md"""
Número de gerações: $(length(fits_11))

Quantidade de avaliações da função objetivo: $(length(fits_11)*length(X0_11))

Valor mínimo encontrado: $(minimum(fits_11[end]))

Valor mínimo da função: $(f11a)
"""

# ╔═╡ 86c88686-aa8a-4ba1-8f28-bc265562203e
let
	x = range(-5,30, length=200)
	y = range(-40,0, length=200)
	xy = map(x->collect(x), Iterators.product(x,y))
	z = f11.(xy)
	
	s = surface(x,y,z;)
	c = contour(x,y,z; fill=true)
plot(s, c; layout=(2,1))
end

# ╔═╡ 2f85fd12-08c5-46d1-8542-8183775f0f25
MD2 = [
	 -6.4985711895798781e-001 -7.6005639589416241e-001
  7.6005639589416241e-001 -6.4985711895798781e-001
  8.9822993821175368e-001 -4.3952585600861893e-001
  4.3952585600861893e-001  8.9822993821175368e-001
 -2.5517719061728122e-001 -9.6689430724804226e-001
  9.6689430724804226e-001 -2.5517719061728122e-001
 -6.4570410476787210e-001  7.6358772193240565e-001
 -7.6358772193240565e-001 -6.4570410476787210e-001
 -8.6934765610621867e-001 -4.9420102470818883e-001
  4.9420102470818883e-001 -8.6934765610621867e-001
  4.4934844932308798e-001  8.9335657555700354e-001
 -8.9335657555700354e-001  4.4934844932308798e-001
  9.3449633491188222e-001  3.5597275181712890e-001
 -3.5597275181712890e-001  9.3449633491188222e-001
  8.9749712493043288e-001 -4.4102030649575180e-001
  4.4102030649575180e-001  8.9749712493043288e-001
  5.8091130819342474e-001  8.1396686174131427e-001
 -8.1396686174131427e-001  5.8091130819342474e-001
  2.4297106046965544e-001 -9.7003353744819110e-001
  9.7003353744819110e-001  2.4297106046965544e-001
]

# ╔═╡ b184e8dd-d960-4d4e-8108-067f561cf88a
M(i) = MD2[i*2-1:i*2,:]

# ╔═╡ db52ac1e-8c29-4858-8422-bd72eb77545c
begin
	f2a = -1300
	
	function f2(X)
	    r = 0
	    D = length(X)
	    Z = Tosz(M(1)*(X))
	    i = 1:D
	    sum(1e6.^((i.-1)./(D.-1)).*Z.^2) .+ f2a
	end
end

# ╔═╡ c5671266-8ac9-45a1-aab8-2337abe20d3c
let
	x = range(-100,100, length=200)
	y = range(-100,100, length=200)
	xy = map(x->collect(x), Iterators.product(x,y))
	z = f2.(xy)
	
	s = surface(x,y,z)
	c = contour(x,y,z; fill=true)
	plot(s, c; layout=(2,1))
end

# ╔═╡ e246d74f-dff3-4a0e-b376-33606a7d3d7b
shift = [-2.1984809693274691e+001  1.1554996930588054e+001 -3.6010680930410572e+001  6.9372732348913601e+001 -3.7608870747492858e+001 -4.8536292149608940e+001  5.3764766904999085e+001  1.3718568644579500e+001  6.9828587467188129e+001 -1.8627811237527567e+001  2.9306608681863466e+001 -7.0216918290093815e+001 -5.1740284602598457e+001  7.1737585569505583e+001 -5.7097788490456374e+001  7.4868392084559218e+001  7.5589061492732164e+000  6.0387714099661700e+001  1.5723311866999682e+001  3.1662383508634154e+001 -4.9340767695180610e+001  5.5037882705394999e+001 -5.2664146736691549e+001 -2.6052382991662000e+001  5.4047889276639125e+001 -7.7471421571386827e+001  6.4605085107437162e+001 -1.7712124964030018e+001 -1.1574279228715504e+001 -4.2591386228491821e+001  1.4099868763503226e+001 -2.0849153638990408e+001  1.2829750891508976e+001 -1.3033429343112887e+001 -3.4228784448817365e+001 -6.0900486247534531e+001  3.6699060076450962e+001 -7.5821585436271661e+001  3.0963033600311711e+001 -3.3284036733990504e+001 -3.8685615461807110e+000  1.5045199483878651e+001  8.8286354912650538e+000 -2.2556601714105376e+001  1.7891609883397138e+001 -2.4669921528691788e+001  4.7683985288383184e+001 -4.9744941398180913e+000  4.7272935138506071e+001  2.5724408972544834e+000  1.7430843896556528e+001  1.5815509351999003e+001 -4.7272484230892992e+001 -5.4257724297510066e+000 -7.6879403580687352e+001  6.6984085027044040e+000 -7.4649508224941684e+001 -5.5175137578537210e+001  5.7354474147230484e+001  7.8820349504005264e+001  5.8594759297326398e+001  1.9431800875092918e+001 -3.6222682001394965e+001 -2.6609959940518930e+001  7.3970419589492494e+001 -1.9261770028345069e+001  7.6143100879025411e+001 -5.0989380655288365e+001  1.7374036903733909e+001  7.7202428260201401e+001  1.9478847587184269e+001  2.0737640522988315e+001 -5.4750896785734326e+001  1.4177423520881108e+001 -5.1776674136467449e+001 -3.6839227137287530e+001 -2.8861436171986210e+001  2.1805381646575558e+001  1.7912651467935330e+001 -1.4408324812697078e+001 -7.9902589820036582e+001  2.6419728279755198e+001  6.7032564640766338e+001  1.0521679763847098e+001  7.8885904619952257e+001 -1.2133872984707189e+001 -2.4501870812413273e+001 -6.7223793807861782e+001  1.7755914473197691e+000 -1.4273826569367456e+001  5.2361092979612415e+001  2.5918140690720026e+001  2.6298182907761774e+001 -4.3449847097785010e+000  1.1198083950067803e+001 -7.9772282790046859e+001 -5.2945388609678645e+001  3.4921984014908659e+001  3.6347758129028037e+001  2.1562145357656860e+001
  5.2517294809079587e+001  3.3851794512807736e+001  5.5104557098557912e+001 -5.5054637528074849e+001  8.5584089839013231e+000 -4.6951174692615908e+001  1.8673238072529916e+001  3.2818102316290812e+001  4.5389818681522023e+000  3.4516179545511960e+001  6.2967409876701048e+001 -3.9818787504282067e+000 -1.2980928833769593e+001 -4.3697829876830568e+001  1.4403660897963112e+001  3.0090011459414107e+001 -4.6677915629295647e+001 -6.7000523103175581e+001 -7.0607390156091526e+001  3.8709975440461285e+001  7.6607937716372291e+001 -5.2609391818446127e+000  5.1008397526345178e+001  4.8534699970410875e+000 -7.3785713134370184e+001  4.4367021532306538e+001  4.4488043800154728e+001  3.3665233354543730e+001 -5.7219905951728606e+001 -1.0836190783193876e+001 -6.8848617747435611e+001  6.4996512994780048e+001  2.6980339897280761e+001 -5.6396267559996240e+001  3.7751575193947041e+001  5.6873637107216297e+001 -3.9560349334474807e+001 -6.2226152665375523e+000  6.4295229391989528e+000  6.5174623067973121e+000  3.1614696437964483e+001  4.5198388654927896e+001  7.4400105769230350e+001  5.4676609312269726e+001  5.4734566343384706e+001  6.2139907331888665e+001 -5.6157545643431121e+001  2.5260131585701554e+001  5.4892651145696220e+001  2.9796504426067017e+001  5.0818538014094671e+001 -4.8777624367550331e+001 -6.5305480000987785e+001  2.1469417845793316e+001 -1.8014789001911240e+001  5.4745999286226315e+001  3.8168961647156323e+001  6.3252317455037513e+001 -3.6076771588706890e+001 -5.2722153595332131e+001 -6.3683256055580451e+001 -1.9144276014621497e+001  3.7041359896598522e+000 -1.4192237166170599e+001 -1.3427327999533333e+001 -5.2005398151226530e+000  5.8438136207298911e+001  7.0587653419956979e+001  4.9747642385394947e+001  6.3372840561342386e+001 -4.0221317731741799e+001  5.3712576303104896e+001 -2.9711884667088754e+001  4.4278752156178214e+001  2.2502446263915818e+001  1.4213321028905666e+001  2.7995571357099273e+001 -3.8768724133617610e+001 -8.1592563585877009e-001 -2.5143814191823825e+001 -2.5079094976284892e+001 -2.3110621502575857e+000 -6.3780203920472424e+001 -2.3955678282236530e+001 -4.4531886686084704e+001  3.2997997780451314e+001 -1.8416536209900727e+000  3.2039888779726070e+001  2.7622659136780296e+001  3.2398684622363660e+000  4.6425436526224878e+001  6.0373080293468277e+001 -6.2001929659800041e+001 -7.9884462956368978e+001  5.4260955790522054e+001  2.2493082083484708e+001  5.4473294185986241e+001 -1.8916909091648677e+001  6.9473150818602079e+001  3.5956298435916061e+001
  1.6623022192495863e+001 -7.3937675412068359e+001 -6.8000746684145355e+001  4.7335018707895760e+000  2.1213325155580261e+001  5.3395098412759140e+001  3.2010072568783229e+001  7.8962215064229838e+001 -9.9300172202720045e+000 -7.2974977674299410e+001  2.7825755937895011e+000  2.8973252620234270e+001  6.5682814603195112e+001  5.8865763633054989e+001  2.8583226731935873e+001  4.0675103085435914e+001 -6.3006312479917490e+001 -2.0503912103201831e+001 -2.1962880434679832e+001 -7.0288055143748124e+001 -4.2455776418032073e+001 -7.9042337868528207e+001  6.4939185236245434e-001  6.4387417511431124e+001  1.1344831560436781e+001 -9.3314346406589976e+000 -3.6939026544271350e+001  6.0833368635549114e+001  5.3275650347420779e+001  2.0817278367812044e+001  1.9061429596630504e+001 -4.7876168311150209e+001 -8.1928201133497573e+000  4.7582972926791307e+001 -4.8233097587699966e+001  7.9683388916261038e+001 -7.9874253320157266e+001 -2.8160027103087877e+001  5.5028377368932354e+001 -1.0515030980969636e+001  4.3170925616812063e+001  6.5166770164353764e+001  5.1618431907532681e+001 -1.4199385719057373e+001 -6.1217827353511236e+000 -6.2452188005801837e+001  7.9171403303182245e+001  5.4403324311841317e+001  8.4841864844992276e+000 -7.8205787325986421e+001 -7.9170286824575342e+001 -1.3572869616031879e+001 -1.5143409879362448e+001 -6.2659418071761600e+001  1.1549484470713184e+001 -7.1301879947408594e+001 -4.5142652066927482e+001 -6.7372032163109878e+001 -7.4793540163130103e+001  1.2038214409167159e+001 -7.3323937630071271e+001 -4.8644502527182794e+001 -6.3816988614644231e+001 -5.9223827656222220e+001  5.2173357377798538e+001  7.1852649856055336e+001  3.3866317993165588e+001  6.0294563167588066e+001 -2.9198852529873115e+001  4.1182946470197599e+001 -5.1481865160379179e+001 -5.2078801770223233e+000  4.3578484195235241e+001 -1.2044828522399007e+001  6.9075374225377345e+001 -2.3147746236261440e+001 -4.7655852396506781e+001 -9.1457557702297549e+000  7.3325112217533189e+001  1.8925708732758110e+001 -1.5010194533222521e+000  5.0993506347452325e+001 -3.0261919128656665e+001 -6.5348749038802055e+001  1.5324369097427297e+001  6.3310618289525067e+001 -6.0088284559631951e+001  5.4289244586114663e+001  4.1634402956413126e+001 -2.1077542078871289e+001 -3.9975992630663512e+001  7.5080259800410658e+001 -3.7721595040094890e+001 -5.7346663201266338e+001  7.9328232413977645e+001 -2.9246022442507972e+001  2.8618215461485839e+001 -3.7501296313004765e+001 -3.1417083089532351e+001 -2.0013270846849483e+001
 -4.0804163037962809e+001 -2.0932733816250877e+001  4.7266808180027951e+001  3.3349339210899927e+001 -5.4463741445113037e+001 -2.2749551391535004e+001 -5.3028532919781650e+000 -1.7275801745638631e+001 -4.4154239667275633e+001 -1.3365989745339098e+001 -3.1599050282534265e+001 -3.8808150203784052e+001 -2.5786013321099063e+001  2.3272397374288536e+000 -3.5982343379027782e+001 -3.9030633732180931e+001 -7.3447336208603218e+001 -3.7248742659316044e+001  6.8912886999507464e+001 -9.2711710369805953e+000 -7.8614063098128867e+001  7.7139347208133259e+001 -3.9009425978886654e+001 -6.1404375381002296e-001 -3.0318221288982187e+001 -2.7821935627808912e+001 -3.0085993994493457e+001 -4.3706846449022066e+001 -6.7509097291658563e+001  6.0185138061179138e+001  5.7868168697322098e+001 -5.6170677178236460e+001 -3.9226442716237358e+001 -1.6502623211700179e+001  6.1733516314151260e+001  1.8233349821775903e+001 -1.1707831813908371e+001  1.0737630238222271e+001 -3.7490987559919844e+001 -7.5370439673534165e+001  4.8095016378163606e+001 -5.2548986802938742e+001  2.6786487497366771e+001  2.5400565183455051e+001  3.8755049663576884e+001  7.8025707425787076e+001 -2.2474392272946531e+001  9.4400047237402926e+000 -1.0569654865341111e+001 -1.4840644693212449e+001 -2.6265030201434541e+001  4.2990102608694158e+001  6.9236561883007639e+001 -1.7271532383621967e+001  4.9567529626486419e+001  4.0152963402019154e+001  5.5848680092236670e+001  3.3859706976384203e+001 -5.9909403544942457e+001  2.9262444185404018e+001 -1.3452172652384927e+001  3.7867449527580590e+001 -7.6221050058228144e+001  6.3096175062610811e+001 -1.6187263838861050e+001  9.1739231725056403e+000 -1.3156053621452907e+001  4.6018330491316945e+001 -1.0676915500583393e+001  2.5124350562151179e+001  5.0414289071995178e+001  1.2812109105593443e+001 -3.7473738844761229e+001 -3.2454754117632632e+001  9.8420671428206354e+000 -7.1715860794521475e+001  4.9014028724310839e+000  3.5727832006214292e+001 -4.1272403933697625e+001  5.6421259869610502e+001  4.9872657941296012e+001 -4.8112051754380268e+001 -4.5814883261774852e+001 -3.4226847116075049e+001  6.3431157480297117e+001 -5.0820970706500418e+001  2.4350567838189374e+001 -3.6285670269475744e+001 -4.2102507600332586e+001 -3.3988787236384773e+001 -2.0470402942193793e+001 -4.3353804471765947e+001 -6.7542497526361728e+001 -2.4564906084717563e+001  4.3471004003682410e+001 -4.9626893520978015e+001  7.9000747926694586e+001 -5.3468668601290084e+001  1.1056708005955580e+001 -3.7930699431497512e+000
  9.6529421914411451e+000 -8.4096517522943905e+000 -5.0972986667719006e+001 -2.8356244364065990e+000 -6.9217682293650611e+001  2.2750744821939961e+001  1.3545909579528896e+000 -3.8785709657205928e+001 -6.3974269215763925e+001  3.0746349702632962e+001  4.0374657525662876e+001  8.5850119192550238e+000  4.0176757187703970e+001  4.5522178388768019e+001 -1.5914631842361882e+000 -2.7849820137133761e+001 -5.6204134787402182e-001  4.5801170915282015e+001 -6.3924291436461900e+000 -3.0603288738486377e+001 -7.3912026580987398e+000  3.4497944264463460e+000 -2.1092516926605938e+001  2.7417226343720049e+001  3.0366194289075086e+001  1.8839617762050935e+001 -6.7843574657181193e+001 -5.0057269006763981e+001  3.6326459016473265e+001  2.4586190079650780e+000  3.2804851359042765e+001  3.4153556210967942e+001 -7.6713594673191523e+001  4.9324327890607442e+001  2.0069225855428012e+001 -5.4572183742658105e+000  4.6767071613428666e+000  6.1389261736339954e+001  3.8670975635949127e+001  1.6830511989675166e+001 -4.5994736585592470e+001 -4.0568683750618803e+001 -1.0233683934434952e+001 -6.3390093348573060e+001 -1.7102905714424985e+001 -3.2829237525515651e+001 -4.1175802565341435e-001 -2.2761202516841809e+001  6.4243493094032814e+001 -2.4053172007478306e+001 -5.5872116984686180e+001  1.8748066464641024e+001  3.0843727432160776e+001  3.3177425949823167e+001 -5.6769482765104115e+001 -6.1058486210723117e+001 -3.1899287996766851e+001 -3.8724554179474566e+001  1.7989589386603310e+001 -2.8414563144964745e+001 -1.6443433190616283e+001  7.0479811635916775e+001  5.2250683439799595e+001 -3.6701421083433587e+001 -4.2808782359449637e+001  4.8271605821545876e+001  1.7967209165465018e-002 -3.0018699231189299e+000 -6.9033271040820779e+001 -4.9283728981324138e+001  7.9024852584473280e+001 -6.0246078273322532e+001  3.0108448192488627e+001 -1.7468334843112785e+001 -1.4563301668544797e+001 -6.0191132948350983e+001  6.4542467734175460e+001  7.2012411228927348e+001 -7.1427991375370553e+001  7.0100709940183464e+001  3.5822718522534132e+001  1.0525208977002885e+001  5.7480063233219184e+001  7.5998626202688129e+001 -1.9647744597550087e+001 -4.3051825660581052e+001  7.2559176278056952e+001 -4.1253291814382589e+000 -6.2410738682311209e+000 -6.1776236147960887e+001  2.6464933700398859e+001 -5.6036165781292752e+001  6.6428277798459675e+001  7.9554910670282169e+001  2.8733321491522815e+001  7.4774377159932868e+001 -1.7329984710233276e+001  7.9276894160319941e+001 -4.9450052843549543e+001  6.7333506713322365e+001
  7.5040483185764245e+001  6.8983059992841802e+001 -1.7436242070098093e+001 -7.0755841310260266e+001 -8.4371913650929464e+000  1.2217775571947145e+001 -5.6608587531638278e+001 -4.9413916738816766e+001  5.8956864221651763e+001  1.4769889536949972e+000  6.7851817818471787e+001 -3.0469868349928483e+001  2.0559410665742977e+001 -6.6589677853696600e+001  5.7573972874039491e+001  4.9159544656690999e+000  5.0418623192354723e+001  1.8604300137081989e+001  5.6980737155742773e+001  3.8024643099227204e+000  4.3233336554914771e+001  1.9710210259005347e+001  7.1907875550453397e+001  6.3671948978276049e+001 -5.8174329773906784e+001  2.9156891935437454e+000 -8.4074844906489332e-001  3.1022458493239185e+001  2.8066557719542438e+001  6.9718595164322565e+001  6.8204247172518706e+001 -6.8671922917680757e+000 -5.6218997152951339e+001  6.7379793686451436e+001 -5.9284847750068419e+000  3.9291353327891329e+001 -6.3824153734750219e+001  7.2513998163135838e+001 -5.7077782532046648e+001 -5.7106767987142604e+001  1.3956425642121966e+001 -2.8587672122269563e+001 -4.5192537652228040e+001  7.9216346735998172e+001  1.5104332759604361e+001  1.9268383173578287e+000  8.7372450308024394e+000  7.9481588186740396e+001 -6.2976132696346525e+001 -4.2651433603347321e+001  2.6891153188675458e+000 -6.4389467944622098e+001  4.9114963700089660e+001 -4.4419615009061189e+001 -3.3889152984312808e+001  3.0977100051288829e+001 -8.9447698542017307e+000 -6.0341701623213826e+000  4.3609078421715203e+001 -4.0935725092218718e+001 -4.1191523639987437e+001  1.1491012492544337e+001 -1.6494854979561911e+001  1.5600116805526897e+001  3.1541580979834144e+001  4.7109075552778620e+001 -3.6565669230799017e+001 -1.9775330586763229e+001  9.2410223073310380e+000  9.2922343191561421e+000 -7.0274365622710832e+001 -2.7818700143351077e+001  1.0227981318957559e+001 -6.5766780515051224e+001  3.4377791256192531e+001  1.9462744462931031e+001 -8.3497814201052272e+000 -2.5541521641260204e+001  6.0948516358776097e+001 -5.7245635868889224e+001  2.8540947083022438e+001  4.7433418502101155e+001  4.7983171910296896e+001  4.5866110978187812e+001 -4.4713726095709472e+000  7.6156115187435447e+001 -7.0209994174851062e+001  7.6610297293171982e+001  7.0190365792482893e+001  1.8814109362609869e+001 -9.7698278382481796e-001  6.9297837965994766e+000 -2.7481496204938043e+001  4.3195036952346832e-001 -3.3028726129186108e+000  3.3992226592244101e+001 -4.7951775875122671e+001  2.7852022903012077e+001 -3.6221437908090188e+001 -3.8506540277753018e+001
 -7.6798583455947181e+001 -3.2208284164745102e+001  2.9633853117889316e+001 -4.7352170508821480e+001  6.1949923589112409e+001 -1.8317739761602478e+000 -6.4734548366169108e+001 -1.4578696307892269e+001 -2.2338893657912642e+001  5.7015693816532817e+001 -9.0048286965158866e+000  6.3352607929054749e+001  7.7040954781927953e+000 -1.7435234983113915e+001  7.7590296507597813e+001 -6.3653351682376694e+001 -2.8736737961965467e+001 -6.0530403377651830e+001 -3.8474686130505575e+001 -5.4006190029887399e+001 -2.7515940326438802e+001  4.6574311272850260e+001 -1.4118903934463539e+001 -4.4545343195892713e+001  4.5324633314592027e+001 -3.6859323453406951e+001  3.1851884590349634e+001 -7.1767596104985515e+001 -2.1194072957097312e+001 -7.3013887047706618e+001 -3.9703351346463030e+001 -7.1267105724160174e+001  5.3914225438710396e+001  2.5131850424594766e+001  6.9098129675000948e+001 -3.7357844526493331e+001  5.1198157306051385e+001  4.7178541320329785e+001  7.7842887643944493e+001  3.3205929656231525e+001 -5.5820954995518598e+001  2.2181559897389221e+001 -7.9961882162038464e+001 -4.4260921387419529e+001 -4.3426076978333640e+001 -7.6934243094844675e+001 -7.0724150501455327e+001 -3.5402173792554180e+001 -2.1492063250907258e+001 -5.5284163202328841e+001 -4.4228714867922648e+001 -4.5363559932288297e+001  3.4330162130131292e+001  7.3395465799305271e+001  3.8953975873385254e+001 -2.8543931397892880e+001  1.8174470868000466e+001  7.5741145297554283e+001  9.8991653462570675e+000  5.9219308022211735e+001  3.5185670724882769e+001 -1.3061653252643000e+001 -1.2906427354640469e+001 -7.8360663514422143e+001  6.8139165005550657e+000 -4.7811305993142007e+001 -2.1903725928412982e+001 -4.3237883786696941e+001 -4.7527049606102132e+001 -3.3009312220655985e+001  3.5904269471244532e+001  6.8660369391101312e+001 -7.2242722911548128e+001  2.7882249352658775e+001 -6.9158442950431521e+001 -7.2805301666001867e+000  3.8130657720736174e+001  5.7758439742176733e+001 -1.9241606927399999e+001 -4.1157421234153460e+001  7.7668720569613569e+001 -6.7013768467955998e+001  2.8983680840861268e+001 -5.4438344652620955e+001 -7.2917070338275835e+001  1.0507297619269881e+001  3.7146771612627177e+001 -4.9970635336254112e+001 -7.9252334634463040e+001 -6.8586236146689615e+001 -7.2098379798238227e+001 -5.4355110032889016e+000 -7.4646023996884150e+000  4.2292475044927080e+001 -6.2182868439626212e+001 -3.2310769230032037e+001  1.3845401073739453e+001  6.0669865503835794e+001 -7.2700771140501701e+001 -4.8078358332953727e+001
 -5.7595941240445747e+001  2.4220045424573406e+001  5.9058122572054392e+000  2.7584588246507636e+001  3.4335584046974375e+001 -6.6588556522001710e+001 -3.4801735669054466e+001  2.2951640505169646e+001 -7.1510608596852450e+001 -4.9933678960621549e+001 -7.0354263927921323e+001  4.5860383614690804e+001 -6.6613311087835157e+001 -4.9869367606876963e+001 -6.4703311321584351e+001  5.0369849815838755e+001  7.4911878266077593e+001  7.3197862540696661e+001  4.5280561809578842e+001  7.9321112476669100e+001  3.3056634177276094e+000 -4.5541439337600380e+001 -7.4099387465168405e+001 -7.5555256007976723e+001 -6.4173725143228317e+000 -4.9998321447326916e+001 -5.1787153319922716e+001  6.9263133736200828e+001 -4.6374286250850318e+001 -1.8242954270417620e+001 -1.6494433930445830e+001  4.8379507474058258e+001 -3.0723224322992650e+001 -7.6207797885410159e+001 -7.3799532948359641e+001 -7.6963201227973698e+001  2.5298986774642383e+001  2.0784304240326378e+001 -6.9685735153873537e+001  5.9226714684901715e+001  7.8951741986742178e+001 -7.4772718111095145e+001  4.1133040694465329e+001 -7.6175966696657895e+001  6.9162628966844480e+001  4.6176096785858050e+001  5.5642228087942193e+001 -4.9403542922712646e+001  3.0960009339214977e+001  7.0195360934052985e+001  7.1244606150209535e+001 -2.5962567696170055e+001 -3.1348386625528665e+001  1.0150689534343886e+001 -7.8720892980830506e+000  7.7132081244145155e+001  7.8164411377699636e+001 -3.0745779624372943e+001  7.1349347368759368e+001 -7.8327688851664419e+001  6.7118664917201173e+001 -4.6619127593811690e+001  3.9343947825016052e+001  7.9046810926065433e+001 -7.8523841945543168e+001 -5.5392995384679821e+001  1.7465691772482977e+001  6.8372132770549561e+000 -6.3301533169193966e+001 -1.5931196178738700e+001  1.1405900742167821e+001 -7.4196348140566201e+001  6.8452945624235156e+001 -5.6311317016889063e+001 -2.2588805489868538e+001  5.8310422311242284e+001 -5.9055180462841726e+001  7.1587136826095152e+000 -5.6381528922208417e+001  4.1568864668894904e+001 -5.8945774323877764e+001  6.7752459644471941e+001 -3.9152506178696438e+000  6.1381873830965766e+001  2.9149979571368767e+001  3.0137592930599446e+001 -4.3532149299140947e+001  1.0059446247607223e+001 -6.0857397860256356e+001  5.7401152988120046e+001  7.2649051525036469e+001  3.8437465172137230e+001  4.3757383786376522e+001  2.3895355809093964e+001 -2.8386879163894658e+001 -1.4137989851635901e+001  4.7528320405793117e+001  9.4404272376355660e+000 -5.6516762153284112e+000 -7.8013318633422870e+001
  4.1892441415438327e+001  5.7370835909285120e+001 -1.6128614385532956e+000 -2.7994503399846785e+001 -2.3205811546035520e+001  7.3571046686488614e+001  7.9062954697270612e+001  6.3933184829148608e+001  2.9838582900331410e+001 -3.3927194145852397e+001 -6.1761322665306892e+001 -5.7435028970980170e+001  4.9996200810238712e+001  2.0194197607994496e+001 -2.6304108954191609e+001 -1.4212305903993702e+001  3.3214857205958936e+001  1.0299339590692664e+001 -5.6356302661624255e+001 -4.4972132854270839e+001  2.6688455658046550e+001 -2.8315688150538353e+001  2.9486300277677131e+001 -6.2269101919259583e+001 -3.9822685589600269e+001  5.3919960777463132e+001  3.5974756921986288e+000  1.5013601450858541e+001  6.7912802962127458e+001 -5.7283112130824541e+001 -1.8200102449554638e+000  2.3772864684963544e+001  6.9334668198903557e+001 -4.7079184492112866e+001 -2.2497593750626098e+001  7.1515216994436583e+000  7.7172935900891801e+001 -5.4451637235617582e+001 -2.1951898617089633e+001  7.8314025313000613e+001 -2.7211156831807404e+001 -7.4740586007804701e+000 -5.4404737407733201e+001  6.3779285782768325e-001 -6.0507273628181082e+001  2.7335119192737181e+001  2.9567581799931119e+001  3.6445097944199603e+001 -3.3807411973873698e+001  4.3806739078756188e+001 -9.2132990880156882e+000  5.0884836579122016e+001  9.6357258489279847e+000  6.0324724314457377e+001  6.6170161026859404e+001 -1.1240982029596278e+000 -6.3394076117204307e+001  3.1717512102611988e+001 -2.8234973618706487e+001  3.3969463728806907e+001  1.9330197221259958e+001 -7.2202784010345781e+001  6.9687995240666737e+001  2.4456035142461921e+001  4.6423323887980814e+001 -7.4525409012646151e+001 -6.1302667947003904e+001  2.4893058962572074e+001  6.5143969868867757e+001 -7.2134666411573846e+001 -1.9531533439351865e+000  4.3122344759147751e+001  6.0293568981816719e+001  6.9635671769166393e+001  5.5345566956111199e+001  4.1665453427943937e+001  5.4000552717197252e+001 -7.4180330790622648e+001  3.9709894460036097e+001 -7.3838597306630916e+001 -4.7232273875702724e+001 -1.8254348559406250e+001  9.9316416492454636e+000  3.0473291730741167e+001  3.6024211104953707e+001 -7.0152325629486228e+001  1.5956681970435431e+001 -2.0676198815329023e+001  5.4875729366164819e+001  7.9683366546585972e+001  1.3588193594496893e+001 -2.3979161048569686e+001  7.3234203266228555e+000  5.0785427579616211e+001 -6.9081177330208192e+001  1.3725832693771306e+001 -6.8709281465208448e+001 -7.7831041935094348e+001  5.1853708972464091e+001  1.0603155755866384e+001
 -1.3830402754516058e-001 -5.8241027810806408e+001  7.0656288551574505e+001  6.0129686491189474e+001  6.8327564260077864e+001  4.2087199328522033e+001 -3.0742718927590278e+001 -7.9516154299190859e+001  3.2776593753620382e+001  6.8096980141002959e+001 -3.6828807497079005e+001  6.8336825880435171e+001 -3.6522458805219060e+001 -3.9576352876368954e+000  4.7412078385892642e+001 -7.5659832380025961e+001  3.0291691868377910e+001 -6.9288431308647684e+000  1.9006519056656476e+001  5.4820288476337787e+001  5.2821440792775157e+001 -5.0503852533248164e+001  4.2812666815669786e+001  4.0430636121919150e+001  7.0173616298122312e+001  7.1250038142402190e+001  6.2355372913445358e+001 -8.1146271000762589e+000  4.7349009377179643e+000  3.3726432336022661e+001 -5.6473161984155780e+001  1.0887072271628012e+001  4.0565123582673309e+001  5.9491681805380869e+000  1.1167287772530905e+001 -2.4849227614675673e+001 -2.4393054809153266e+001 -4.7476223043430537e+001 -1.5507046249355399e+001 -2.4889459940419115e+001 -6.7444686649800630e+001  6.3663664722148283e+001 -1.8569526111201625e+001  3.2466337052501046e+001 -6.8880401283241554e+001 -6.2007704554856140e+000 -4.0463824493807607e+001 -7.5734196194535528e+001 -6.8655843007478410e+001  6.3001811073656640e+001  4.0200464483038452e+001  7.1644821478752988e+001 -4.9767791365600772e+001 -7.4296501286098220e+001  2.4778130721734193e+001 -4.2281185676379792e+001  5.4249903551549288e+000  5.0935165274886165e+000 -9.5201130942729542e+000 -6.5851784818042969e+000  8.2573266794503191e+000  5.8821643963427604e+001  1.9695440839281122e+001  3.4488792263096194e+001 -5.8424943874158927e+001  2.2644749555071897e+001 -6.8365285184222984e+001 -6.6613719875982753e+001  3.3217875037889961e+001 -1.8332999781028988e+001 -1.9089037678932996e+001 -4.0167910518819241e+001 -1.3392391893373789e+001  5.0142665924105586e+001 -4.0143269959076250e+001  6.6843942525311505e+001 -7.8303967128687333e+001 -5.9097408181889826e+001  1.4102125260795574e+001  3.3289099896209713e+000  9.9774918580544387e+000 -4.0157458286387509e+001 -7.8961112136672710e+001 -2.5036725043570129e-002 -6.3346527624019849e+001 -1.6216447255667607e+001  6.2315240934716648e+001  2.4487572294902265e+001 -2.6152984830057417e+001  4.3478485426115981e+001 -6.1916386327020071e+001 -7.0379944991649467e+001  6.3947834210623590e+001 -3.4170395657495312e+001 -3.6255956670435822e+001  5.0612215320021853e+001 -1.5043517406481929e+001 -7.0832508057556378e+000  2.1595077007462290e+001  6.3968505893561989e+001]

# ╔═╡ 44589478-3a1e-455a-b218-2025451d5111
o(i, N) = shift[i,1:N]

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
IterTools = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Combinatorics = "~1.0.2"
Distributions = "~0.25.34"
IterTools = "~1.3.0"
Plots = "~1.24.2"
PlutoUI = "~0.7.21"
StatsBase = "~0.33.13"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "7f3bec11f4bcd01bc1f507ebce5eadf1b0a78f47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.34"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "74ef6288d071f58033d54fd6708d4bc23a8b8972"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "93f484f18848234ac2c1387c7e5263f840cdafe3"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.24.2"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "b68904528fd538f1cb6a3fbc44d2abdc498f9e8e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.21"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═6021fa5e-d35c-4311-a83c-ac33c665ab02
# ╟─0091b053-f24e-406c-9b48-04882356ad86
# ╟─3475316d-8524-430d-918e-ece3a5b76bc9
# ╟─bba34a41-77db-4225-966d-fabd58896074
# ╟─d0ceb131-98ea-4a6d-979f-42bd0bbeecb6
# ╟─53e4ca4f-4ead-40fd-b6c1-6bdc63352a6b
# ╟─1a39b939-10f9-4ad3-ac30-3bc2d6934071
# ╟─07054dd1-386e-4deb-8ff8-3ef912d43d98
# ╠═c1ba8e98-6a4e-45e6-8fcb-4cde49be7fac
# ╟─35af2c60-74d7-4719-a6ca-9124e6e95367
# ╠═e66bff7f-78a1-4317-b7f4-e8287d7a0875
# ╠═8be18e7b-a996-46e7-946f-ceeb82de8bd1
# ╠═f59946ea-c0d6-404c-b453-21e350d9e039
# ╟─88af6f31-2303-4fa8-b5b3-4d36fb9492ab
# ╠═f8c79585-33aa-4627-bf2d-8deebd9ca779
# ╟─bcf5aa9e-30a1-40af-b1ed-2b18c6534e6b
# ╠═5b4588fa-c73b-49b8-a3ec-9b0b30259f40
# ╟─eabccb4b-9890-428c-9d43-dbab84fd08cc
# ╟─f95a1f72-8333-485f-8756-e52187557a78
# ╟─b6790a2a-bc7a-4e16-ab8c-e998d2af5c31
# ╟─3699b074-1242-4a35-bee2-40668bcfd5d7
# ╟─6fbc7788-4736-4346-a08b-a0e0e99f363e
# ╠═2c217353-803a-4556-a4dc-1cdff404e7be
# ╠═0ffcf586-9efb-479a-b984-2b89e3292cba
# ╠═5b6f6cf4-92bb-4a8e-847e-8f7ed3a4e53d
# ╟─d6adfd3c-258e-4cb9-ade9-bf31d0e74b19
# ╠═40e315bf-49fb-4e80-91c2-5ee237c08d0a
# ╟─e598a7e2-a059-47a3-bee5-23890fc4994b
# ╟─7b4128c7-fff5-4bf0-b673-46a7ebc818dd
# ╟─cd96b5e2-d4ae-4b25-aecd-efc02ee96f49
# ╟─75ecda46-8672-4a2d-a051-28132373ab23
# ╟─c305594a-a65d-4c2f-8177-99a334cbebd6
# ╟─fc1541fc-892c-49a4-80e0-3829c3dde0d7
# ╟─6fe88ef4-0342-4632-bb98-e3e36e2181e4
# ╠═c0ad6d28-4046-47cc-9ae6-6012b7f21ce9
# ╠═cf8b1a3f-70d5-490f-83bb-881fe73c0c16
# ╠═0e6096a2-28b5-42a0-9ca6-c01268e1b28f
# ╠═b13ba3ef-f838-4758-ae04-59e8be85e250
# ╠═f162da3d-0165-4417-9f88-0ceb31869f88
# ╟─d9dec120-048b-49fc-9684-ce28c69a56e1
# ╟─23a16641-13e5-4a35-895e-8752c9d4e0dd
# ╟─0360fb13-f186-40a4-9ee6-cf7fb80dd659
# ╠═e1464a65-80b2-415b-9ab2-5547edb12f74
# ╠═4dab05b9-b1b2-453e-8f2f-09c1632b3d48
# ╠═c9a7c598-230b-41fa-a992-747c7e640da9
# ╠═60fe54f6-837e-4cac-903a-3db308f71d8f
# ╠═2b2509dd-2ed6-4f9f-9a28-a273d44fe5ea
# ╠═e1cc07c2-d7d0-4344-8f32-a8b49a357e4b
# ╠═4546f9d3-1f6e-4a04-9bc6-8eaf44c4f7eb
# ╠═c9300ca8-205c-44ce-a74d-bc1af03a8a48
# ╠═2a85f4f2-91c8-4e58-a06c-c80cb4b0d7fe
# ╠═ca3796ef-2c3e-486b-b571-d17a278ad1c9
# ╠═63364c03-04db-414b-a58b-c057da38166e
# ╠═4b3b752b-54c1-44ff-baba-232a0a57ff08
# ╟─53044516-2a6f-433c-b2aa-5855a02009c1
# ╠═d25316bb-a1cb-49e6-bf1b-0bfd7b678791
# ╠═7d4845c3-2043-44cd-83c6-bcebf0a01ea2
# ╠═c1692fd0-7154-4757-8e78-01d99795a0e4
# ╠═57b6b893-bd08-4b54-ba77-efb1484a768b
# ╟─707c0054-5bed-4909-bd2c-f2392948ca0b
# ╠═656f7acb-5b15-44d9-b225-074280b597ea
# ╠═c2a65bb4-ff08-4f0b-ade7-a2a2800bf1cc
# ╟─f6c638ba-0248-4b88-8dce-e0c35608a092
# ╟─761370a2-7cb3-4142-8845-f1bb3fa3b195
# ╟─c18e2efb-8710-4a67-9689-ede1fe877b2d
# ╟─e6b953f3-1d3d-4a45-a928-6ee8e283b588
# ╟─38318598-22c4-49a2-900e-6d63fc94add0
# ╠═8238c94e-0c62-4cd3-bbc5-b33f08c30914
# ╟─dd586276-6afe-4016-beb0-fe1bc59b7fb5
# ╟─ae6c7f08-6f7d-49e7-9825-1c4d69dea2dd
# ╠═db52ac1e-8c29-4858-8422-bd72eb77545c
# ╟─c5671266-8ac9-45a1-aab8-2337abe20d3c
# ╟─977cc41a-a5c3-4c63-97b0-752b79a8b13e
# ╠═a235d168-76cb-4c1e-8d72-80b55a13b97d
# ╟─86c88686-aa8a-4ba1-8f28-bc265562203e
# ╟─d7d5e1b9-3082-4e25-a2d3-8d0e61758289
# ╟─885f143e-4708-4c37-9cac-0cf99b4f0682
# ╟─bb8e4d4b-04e1-4566-bf96-4860fa4e2735
# ╟─1b1c87ab-9c57-4c6e-9651-b0fc58a352ca
# ╟─6cf2bcc2-0ca3-4946-adaa-21f6c700ccb6
# ╟─db6af83c-fc12-43f6-9a4b-459fb659d132
# ╟─a0092fbe-c792-4e92-b5e3-ad79ef77f5be
# ╟─9838e06b-a30f-426f-b319-51bcf54d45d7
# ╟─03cf431e-3d6c-4167-81bf-9df45ce6182b
# ╟─44589478-3a1e-455a-b218-2025451d5111
# ╟─b184e8dd-d960-4d4e-8108-067f561cf88a
# ╟─5dd415b1-ae92-4e59-9fc5-7060dde228ab
# ╟─885400f7-32af-42d6-b7a7-68000228263b
# ╟─a9f4ec4e-4590-4114-a920-11b2654e0991
# ╟─57a57096-97a9-4ead-97bb-03cc5dcf6bd7
# ╟─19d4d957-b1db-4c09-b808-3eee5463ff68
# ╟─78f05a6c-c7f9-450a-ae86-3b1777c89dc3
# ╟─68738ecd-2769-4b8a-be9b-a138745ca829
# ╟─2f85fd12-08c5-46d1-8542-8183775f0f25
# ╟─e246d74f-dff3-4a0e-b376-33606a7d3d7b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
