using StaticArrays
using LinearAlgebra
using Kepler
using SPICE
using CairoMakie
using Printf
using ProgressMeter
using Dates
using CSV
using DataFrames
using DataInterpolations
# using Optimization
using AstroTime

# TODO: send launch 

furnsh("data/naif0012.tls")
furnsh("data/de440.bsp")
# furnsh("data/2024_PDC25a-s3-merged-DE441.bsp")

function period(orbit::Kepler.Cometary)
    @assert orbit.e < 1
    a = orbit.q/(1.0 - orbit.e)
    return 2pi*sqrt(a^3/orbit.gm)
end

function spice_lookup(id, ref, mjd) 
    # et = SPICE.str2et("JD $(mjd + 2400000.5) TDB")
    et = str2et(@sprintf("JD %.6f TDB", mjd + 2400000.5))
    return spkezr(id, et, "J2000", "None", ref)[1]
end

# load asteroid trajectory interpolant from a grss integration dumped to a csv
# NOTE: Barycentric coordinates
ast_tbl   = CSV.read("data/pdc2025_traj.csv", DataFrame)
interp_x  = CubicHermiteSpline(ast_tbl[:, "vx"], ast_tbl[:, "x"], ast_tbl[:, "mjd_tdb"])
interp_y  = CubicHermiteSpline(ast_tbl[:, "vy"], ast_tbl[:, "y"], ast_tbl[:, "mjd_tdb"])
interp_z  = CubicHermiteSpline(ast_tbl[:, "vz"], ast_tbl[:, "z"], ast_tbl[:, "mjd_tdb"])
interp_vx = CubicSpline(ast_tbl[:, "vx"], ast_tbl[:, "mjd_tdb"])
interp_vy = CubicSpline(ast_tbl[:, "vy"], ast_tbl[:, "mjd_tdb"])
interp_vz = CubicSpline(ast_tbl[:, "vz"], ast_tbl[:, "mjd_tdb"])
ast_interp(mjd) = (interp_x(mjd), interp_y(mjd), interp_z(mjd), interp_vx(mjd), interp_vy(mjd), interp_vz(mjd))

# constants
au2km   = 1.495978707e8
gm_sun  = 0.01720209894846^2
day2sec = 86400.0 
grav_const = 6.6743 * 10^(-11) # m kg s

r_ast = 150/2 # m
m_ast = 3.97e9 # kg
E_bind = 3*grav_const*m_ast^2/5/r_ast # kg m^2 / s^2

sqrt(E_bind / 100) / 1000 # km/s

# mjd = 0, for converting et -> mjd
mjd_ref    = str2et("JD 2400000.5 TDB")/day2sec

# approx impact date 
mjd_impact = str2et("2041-04-28T00:00:00")/day2sec - mjd_ref

# for year in 2028:2:2036
# for year in 2030:2030
year = 2030
    et0 = str2et("$year-01-01T00:00:00")
    etf = str2et("$(year+1)-12-31T00:00:00")
    # et_step = 2*day
    mjd0 = et0/day2sec - mjd_ref
    mjdf = etf/day2sec - mjd_ref

    depart_mjd = mjd0:2:mjdf
    tof_days   = 0.0:2:1000.0
    
    mshape = (length(depart_mjd), length(tof_days))
    toa               = zeros(Float32, mshape) .+ Inf
    impact_depart_dv  = Matrix{SVector{3, Float64}}(undef, mshape)
    impact_arrive_dv  = Matrix{SVector{3, Float64}}(undef, mshape)

    data_path = "/Users/samuelcornwall/school/courses/AE498-pd/data/"
    io = open(joinpath(data_path, "$(year)_launch_candidates.csv"), "w")
    @printf io "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "departure_mjd_tdb" "arrival_mjd_tdb" "dV_au_day" "V0_x_au_day" "V0_y_au_day" "V0_z_au_day" "Vf_x_au_day" "Vf_y_au_day" "Vf_z_au_day"
    @showprogress for ((i, _mjd0), (j, _tof)) in collect(Iterators.product(enumerate(depart_mjd), enumerate(tof_days)))
        _mjdf  = _mjd0 + _tof
        state0 = spice_lookup("earth", "Sun", _mjd0)
        # statef = spkezr(id,      _etf, "J2000", "none", "Sun")[1]
        statef = ast_interp(_mjdf)

        pos0 = state0[1:3] ./ au2km
        vel0 = state0[4:6] ./ au2km .* day2sec

        # posf = statef[1:3] ./ au
        # velf = statef[4:6] ./ au .* 86400
        baryf = spice_lookup("SSB", "Sun", _mjdf) ./ au2km
        baryf[4:6] .*= day2sec

        posf = [statef[1:3]...] .- baryf[1:3]
        velf = [statef[4:6]...] .- baryf[4:6]

        toa[i, j] = _mjdf
        lambert_solutions = collect(Kepler.lambert(pos0, posf, _tof, gm_sun))

        dv = Inf
        # k = argmin(v -> norm(v[1] - vel0), lambert_solutions)
        for (v1t, v2t) in lambert_solutions
            # if norm(v1t - vel0) + norm(velf - v2t) < dv
            dv_dep = norm(v1t - vel0)
            dv_enc = norm(velf - v2t)

            if dv_enc < dv
                # dv = norm(v1t - vel0) # departure cost
                dv = dv_enc
                impact_depart_dv[i, j] = (v1t - vel0)
                impact_arrive_dv[i, j] = (velf - v2t)
            end
        end

        if dv < (10 / au2km * day2sec)
            dv1 = impact_depart_dv[i, j]
            dv2 = impact_arrive_dv[i, j]
            @printf io "%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" _mjd0 _mjdf norm(dv1) dv1... dv2...
        end
    end

    close(io)

    tta = (mjd_impact .- toa) ./ 365
    # x   = collect(mjd_impact .- depart_mjd) ./ 365 # (et_impact .- ets) ./ day ./ 365
    # y   = tof_days ./ 365
    
    x = depart_mjd
    y = tof_days

    levels = 0:1:15
    sublevels = 0:0.2:15

    # candidate_depart = 62675.001
    # candidate_tof    = 871

    candidate_depart = 62616.0
    candidate_tof    = 871.0

    for (title, _, M) in [
        (
            "intercept", 
            "ΔV (km/s)",  
            norm.(impact_arrive_dv) * au2km / day2sec,

        ),
        (
            "departure", 
            # "C₃ (km/s)²", 
            "ΔV (km/s)",
            # norm.(impact_depart_dv) .^ 2, 
            norm.(impact_depart_dv) * au2km / day2sec, 
        ),
        (
            "Total",
            "ΔV (km/s)",
            (norm.(impact_depart_dv) .+ norm.(impact_arrive_dv)) * au2km / day2sec,
        )
    ]
        f = Figure(size = (1000, 800), fontsize = 20)
        ax = Axis(f[1, 1]; 
            # xlabel = "Departure time (years before impact)", 
            xlabel = "Departure date (mjd)", 
            ylabel = "Time of flight (days)", 
            # xreversed = true, 
            title = title
        )

        clabel = "ΔV (km/s) (lowest cost branch)"

        # 
        hm = contourf!(ax, x, y, M; levels = levels, colormap = :plasma)

        contour!(x, y, M; levels = sublevels, linewidth = 0.5, color = :black, linestyle = :dot)

        Colorbar(f[1, 2], hm; label = clabel)
        # xlims!(ax, maximum(x), minimum(x))
        xlims!(ax, minimum(x), maximum(x))
        ylims!(ax, 0, maximum(y))

        scatter!(ax, candidate_depart, candidate_tof; marker = :star5, markersize = 20.0,  label = "chosen trajectory")

        ax.yticks = 0:200:1000

        contour!(ax, x, y, tta; color = :black, linewidth = 1.0, levels = 0:0.5:15, linestyle = :dash, labels = true, labelsize = 20)

        axislegend(ax)

        f
        save("project_plots/$year-$title.png", f)
    end

    # impact velocity, compare to disruption limit
    state1 = spice_lookup("earth", "Sun", candidate_depart) ./ au2km
    pos1 = state1[1:3]
    vel1 = state1[4:6] .* day2sec

    baryc2 = spice_lookup("SSB", "Sun", candidate_depart) ./ au2km
    baryc2[4:6] .*= day2sec

    state2 = ast_interp(candidate_depart + candidate_tof) .- baryc2
    pos2   = state2[1:3]
    vel2   = state2[4:6]

    earth2 = spice_lookup("earth", "Sun", candidate_depart + candidate_tof)[1:3] ./ au2km

    rad2deg(Kepler.vec_angle(earth2, pos2))

    TDBEpoch(candidate_depart*days; origin = :modified_julian)
    TDBEpoch((candidate_depart + candidate_tof)*days; origin = :modified_julian)

    lambert_solution = collect(Kepler.lambert(pos1, pos2, candidate_tof, gm_sun))
    i_min = argmin(i -> norm(lambert_solution[i][2] - vel2), eachindex(lambert_solution))
    # i_min = argmin(i -> norm(lambert_solution[i][1] - vel1), eachindex(lambert_solution))
    (v1t, v2t) = lambert_solution[i_min]

    v_inf = norm(v1t - vel1) .* au2km / day2sec
    v_imp = norm(vel2 - v2t) .* au2km / day2sec

    rad2deg(Kepler.vec_angle(vel2, v2t))

    v_lim = sqrt(E_bind/1000)/1000

    v_imp / v_lim

    begin
    @printf "departure time (MJD TDB)    : %.6f\n" candidate_depart
    @printf "departure relative velocity : [%.6f, %.6f, %.6f] km/s (total = %.6f km/s)\n" ((v1t - vel1) .* au2km / day2sec)... v_inf

    @printf "arrival time (MJD TDB)      : %.6f\n" candidate_depart + candidate_tof
    @printf "arrival relative velocity   : [%.6f, %.6f, %.6f] km/s (total = %.6f km/s)\n" ((vel2 - v2t) .* au2km / day2sec)... v_imp
    end

# end
