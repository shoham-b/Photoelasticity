import numpy as np
import scipy


# xxi = x-position at which the stress is computed (in meters)
# xxj = y-position at which the stress is computed (in meters)
# z = number of forces
# f list of forces (magintude) acting on the particle
# alpha list of angles of the forces, see diagram in J.Puckett, PhD-Thesis, North Carolina State University
# beta list of agles of the forces, see diagram in J.Puckett, PhD-Thesis, North Carolina State University
# fsigma photoelastic stress coefficient
# rm radius of the disk (in meters)
def stress_engine(xxi, xxj, z, f, alpha, beta, fsigma, rm):
    # Computes the optical response of any given point in a loaded photoelastic disk
    # result = photoelastic response (intensity)
    beta = -beta + np.pi / 2
    pioverfsigma = np.pi / fsigma
    oneoverpirm = 1 / (np.pi * rm)
    twooverpi = 2 / np.pi

    sigmaxx = 0
    sigmayy = 0
    sigmaxy = 0

    for k in range(z):
        b = beta[k] + np.pi / 2  # rotate pi/2, to match input image
        a = alpha[k]
        if a < 0:
            b2 = b + (np.pi + 2 * a)
        else:
            b2 = b - (np.pi - 2 * a)

        x1 = rm * np.sin(b)
        y1 = rm * np.cos(b)
        x2 = rm * np.sin(b2)
        y2 = rm * np.cos(b2)

        ch0 = x2 - x1  # chord x
        ch1 = y2 - y1  # chord y
        chn = np.sqrt(ch0 ** 2 + ch1 ** 2)
        ch0 /= chn  # normalize chord x
        ch1 /= chn  # normalize chord y

        r10 = xxi - x1  # r vector x coord
        r11 = -xxj - y1  # r vector y coord
        r1n = np.sqrt(r10 ** 2 + r11 ** 2)
        costh1 = (r10 * ch0 + r11 * ch1) / r1n

        if r11 * ch0 > r10 * ch1:  # important!
            signth = 1
        else:
            signth = -1

        th1 = signth * np.arccos(costh1)  # faster than cos(asin(stuff))
        s1 = -(f[k] * twooverpi) / r1n * costh1
        th = th1 - beta[k] - alpha[k]  # rotate coordinates

        sigmaxx += s1 * (np.sin(th) ** 2)
        sigmayy += s1 * (np.cos(th) ** 2)
        sigmaxy += 0.5 * s1 * (np.sin(2 * th))

    aa = np.real(np.sqrt((sigmaxx - sigmayy) ** 2 + 4 * (sigmaxy) ** 2))
    result = (np.sin(pioverfsigma * aa)) ** 2  # wrap

    if np.isnan(result):  # for some reason result sometimes gets to be NAN
        result = 0  # temporary fix is to set it zero then
    if result < 0:  # for some reason, not sure why this happens
        result = 0  # temporary fix is to set it zero then
    return result


def err_from_fit(measured_brightness_arr, forces, alpha, beta, fsigma, radius_px, radius_m):
    # inputs:
    # the measured brightness of the disc, as a square array
    # the force magnitudes (array), and the angles describing their position and direction (alpha, beta).
    # (see Fig. 4.11 of James Puckett's PhD thesis)
    # the fsigma constant (related to C and lambda), the radius of the disc in pixels, and in meters
    z = forces.shape[0]  # the num of forces
    sq_sum = 0.0
    for i in range(radius_px * 2):  # loop over all points in
        for j in range(radius_px - i, radius_px + i + 1):
            x = (-1 + i / float(radius_px)) * radius_m
            y = (1 - j / float(radius_px)) * radius_m
            calc_brightness = stress_engine(x, y, z, forces, alpha, beta, fsigma, radius_m)
            sq_sum += (measured_brightness_arr[i][j] - calc_brightness) ** 2
    return sq_sum


def find_forces(measured_brightness_arr, forces0, alpha0, beta, fsigma, radius_px, radius_m):
    # note: radius_px is allowed to be a float. we can also take it as the size of measured_brightness_arr

    z = forces0.shape[0]

    def func(forces_and_alphas, z):
        forces = forces_and_alphas[0:z]
        alpha = forces_and_alphas[z + 1:2 * z]
        return err_from_fit(measured_brightness_arr, forces, alpha, beta, fsigma, radius_px, radius_m)

    max_force = 1e5
    minbounds = tuple([0.0] * (2 * z))
    maxbounds = [max_force] * z
    maxbounds.append([2 * np.pi] * z)
    maxbounds = tuple(maxbounds)

    result = scipy.optimize.minimize(func, np.concatenate(forces0, alpha0), args=(z), bounds=(minbounds, maxbounds))
    ###** can add arguments for convergence here

    forces_final = np.array(result.x[0:z])
    alphas_final = np.array(result.x[z + 1:2 * z])
    normal_forces = forces_final * np.cos(alphas_final)
    tangent_forces = forces_final * np.sin(alphas_final)
    return [forces_final, normal_forces, tangent_forces]
