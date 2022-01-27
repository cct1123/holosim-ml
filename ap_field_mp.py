"""
Miscellaneous aperture field analysis functions.

Grace E. Chesmore
2021
Chun Tung Cheung 
Jan 2022
"""


import numpy as np
from scipy import optimize
import multiprocessing as mp

import pan_mod as pm
from pan_mod import *
from tele_geo import *

y_cent_m1 = -7201.003729431267

adj_pos_m1, adj_pos_m2 = pm.get_single_vert_adj_positions()

class RayMirrorPts():
    def __init__(self, P_rx, tele_geo, theta, phi):
        theta,phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)
        
        # Read in telescope geometry values
        th2 = tele_geo.th2
        focal = tele_geo.F_2

        n_pts = len(theta)
        out = np.zeros((6, n_pts))

        # class instance variables
        self.P_rx = P_rx
        # self.tele_geo = tele_geo
        self.theta = theta
        self.phi = phi
        self.th2 = th2
        self.focal = focal
        self.n_pts = n_pts
        self.out = out

    def trace_rays(self, ii):
        P_rx = self.P_rx
        theta = self.theta
        phi = self.phi
        th2 = self.th2
        focal = self.focal

        # Define the outgoing ray's direction
        th = theta[ii]
        ph = phi[ii]
        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] in telescope r.f.
        x_0 = P_rx[0]
        y_0 = P_rx[1]
        z_0 = P_rx[2]

        # Use a root finder to find where the ray intersects with M2
        def root_z2(t):

            # Endpoint of ray:
            x = x_0 + alpha * t
            y = y_0 + beta * t
            z = z_0 + gamma * t

            # Convert to M2 r.f.
            xm2, ym2, zm2 = tele_into_m2(x, y, z)

            # Z of mirror in M2 r.f.
            z_m2 = z2(xm2, ym2)
            return zm2 - z_m2

        t_m2 = optimize.brentq(root_z2, focal + 1e3, focal + 13e3)

        # Endpoint of ray:
        x_m2 = x_0 + alpha * t_m2
        y_m2 = y_0 + beta * t_m2
        z_m2 = z_0 + gamma * t_m2
        P_m2 = np.array([x_m2, y_m2, z_m2])

        ########## M2 r.f ###########################################################

        x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(P_m2[0], P_m2[1], P_m2[2])
        x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(x_0, y_0, z_0)

        # Normal vector of ray on M2
        norm = d_z2(x_m2_temp, y_m2_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        # Normalized vector from RX to M2
        vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
        tan_rx_m2 = vec_rx_m2 / dist_rx_m2

        # Vector of outgoing ray
        tan_og = tan_rx_m2 - 2 * np.dot(np.sum(np.dot(tan_rx_m2, N_hat)), N_hat)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)

        N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
        N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
        N_hat_z_temp = N_hat[2]

        N_hat_t[0] = N_hat_x_temp
        N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
        N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)

        tan_rx_m2_t = np.zeros(3)

        tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(np.pi)
        tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(np.pi)
        tan_rx_m2_z_temp = tan_rx_m2[2]

        tan_rx_m2_t[0] = tan_rx_m2_x_temp
        tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(th2)
        tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(th2)

        tan_og_t = np.zeros(3)
        tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
        tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
        tan_og_z_temp = tan_og[2]

        tan_og_t[0] = tan_og_x_temp
        tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
        tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)

        ########## Tele. r.f ###########################################################

        # Vector of outgoing ray:
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        # Use a root finder to find where the ray intersects with M1
        def root_z1(t):

            # Endpoint of ray:
            x = P_m2[0] + alpha * t
            y = P_m2[1] + beta * t
            z = P_m2[2] + gamma * t

            # Convert to M1 r.f.
            xm1, ym1, zm1 = tele_into_m1(x, y, z)

            # Z of mirror in M1 r.f.
            z_m1 = z1(xm1, ym1)
            return zm1 - z_m1

        t_m1 = optimize.brentq(root_z1, 50, 22000)

        # Endpoint of ray:
        x_m1 = P_m2[0] + alpha * t_m1
        y_m1 = P_m2[1] + beta * t_m1
        z_m1 = P_m2[2] + gamma * t_m1

        # Write out
        return [x_m2, y_m2, z_m2, x_m1, y_m1, z_m1]

    def output(self):
        if self.n_pts > 3000: # a threshold to use parallelism (it's a empirical number tested with Ryzen 5 3600)
            pool = mp.Pool(max(1, mp.cpu_count()-1))
            # result = list(pool.imap(self.trace_rays, range(self.n_pts)))
            result = pool.map(self.trace_rays, range(self.n_pts))
            pool.close()
            pool.join()
        else:
            result = []
            for ii in range(self.n_pts):
                result.append(self.trace_rays(ii))

        self.out[0:6] = np.array(result).transpose()
        return self.out

class AperatureFieldsFromPanelModel():
    def __init__(self, panel_model1, panel_model2, 
                 P_rx, tele_geo, theta, phi, rxmirror
                 ):
        theta, phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)

        th2 = tele_geo.th2
        z_ap = tele_geo.z_ap * 1e3
        horn_fwhp = tele_geo.th_fwhp
        focal = tele_geo.F_2
        # Step 1:  grid the plane of rays shooting out of receiver feed
        N_linear = tele_geo.N_scan
        col_m2 = adj_pos_m2[0]
        row_m2 = adj_pos_m2[1]
        x_adj_m2 = adj_pos_m2[4]
        y_adj_m2 = adj_pos_m2[3]
        col_m1 = adj_pos_m1[0]
        row_m1 = adj_pos_m1[1]
        x_adj_m1 = adj_pos_m1[2]
        y_adj_m1 = adj_pos_m1[3]

        x_panm_m2 = np.reshape(
            rxmirror[0, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
        )
        y_panm_m2 = np.reshape(
            rxmirror[2, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
        )
        x_panm_m1 = np.reshape(
            rxmirror[3, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
        )
        y_panm_m1 = np.reshape(
            rxmirror[4, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
        )

        pan_id_m2 = identify_panel(x_panm_m2, y_panm_m2, x_adj_m2, y_adj_m2, col_m2, row_m2)
        pan_id_m1 = identify_panel(
            x_panm_m1, y_panm_m1 - y_cent_m1, x_adj_m1, y_adj_m1, col_m1, row_m1
        )

        row_panm_m2 = np.ravel(pan_id_m2[0, :, :])
        col_panm_m2 = np.ravel(pan_id_m2[1, :, :])
        row_panm_m1 = np.ravel(pan_id_m1[0, :, :])
        col_panm_m1 = np.ravel(pan_id_m1[1, :, :])

        # Step 2: calculate the position + local surface normal for the dish
        n_pts = len(theta)
        out = np.zeros((17, n_pts))
        out[4, :] = y_cent_m1

        # class instance variables
        self.panel_model1 = panel_model1
        self.panel_model2 = panel_model2
        self.P_rx = P_rx
        self.tele_geo = tele_geo
        self.theta = theta
        self.phi = phi
        self.row_panm_m1 = row_panm_m1
        self.col_panm_m1 = col_panm_m1
        self.row_panm_m2 = row_panm_m2
        self.col_panm_m2 = col_panm_m2

        self.theta_m = np.mean(theta)
        self.phi_m = np.mean(phi)
        self.th2 = th2
        self.z_ap = z_ap
        self.horn_fwhp = horn_fwhp  # Full width half power [rad] of source
        self.focal = focal

        self.n_pts = n_pts
        self.out = out

    def trace_rays(self, ii):
        panel_model1 = self.panel_model1
        panel_model2 = self.panel_model2
        P_rx = self.P_rx
        tele_geo = self.tele_geo
        theta = self.theta
        phi = self.phi

        row_panm_m1 = self.row_panm_m1
        col_panm_m1 = self.col_panm_m1
        
        row_panm_m2 = self.row_panm_m2
        col_panm_m2 = self.col_panm_m2
        

        theta_m = self.theta_m
        phi_m = self.phi_m
        th2 = self.th2
        z_ap = self.z_ap
        horn_fwhp = self.horn_fwhp
        focal = self.focal

        i_row = row_panm_m2[ii]
        i_col = col_panm_m2[ii]
        i_panm = np.where((panel_model2[0, :] == i_row) & (panel_model2[1, :] == i_col))

        if len(i_panm[0]) != 0:

            th = theta[ii]
            ph = phi[ii]
            r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

            alpha = r_hat[0]
            beta = r_hat[1]
            gamma = r_hat[2]

            # Receiver feed position [mm] (in telescope reference frame):
            x_0 = P_rx[0]
            y_0 = P_rx[1]
            z_0 = P_rx[2]

            a = panel_model2[2, i_panm]
            b = panel_model2[3, i_panm]
            c = panel_model2[4, i_panm]
            d = panel_model2[5, i_panm]
            e = panel_model2[6, i_panm]
            f = panel_model2[7, i_panm]
            x0 = panel_model2[8, i_panm]
            y0 = panel_model2[9, i_panm]

            def root_z2(t):
                x = x_0 + alpha * t
                y = y_0 + beta * t
                z = z_0 + gamma * t
                xm2, ym2, zm2 = tele_into_m2(
                    x, y, z
                )  # Convert ray's endpoint into M2 coordinates

                if P_rx[2] != 0:
                    z /= np.cos(np.arctan(1 / 3))
                xm2_err, ym2_err, zm2_err = tele_into_m2(
                    x, y, z
                )  # Convert ray's endpoint into M2 coordinates

                x_temp = xm2_err * np.cos(np.pi) + zm2_err * np.sin(np.pi)
                y_temp = ym2_err
                z_temp = -xm2_err * np.sin(np.pi) + zm2_err * np.cos(np.pi)

                xpc = x_temp - x0
                ypc = y_temp - y0

                z_err = (
                    a
                    + b * xpc
                    + c * (ypc)
                    + d * (xpc ** 2 + ypc ** 2)
                    + e * (xpc * ypc)
                )
                z_err = z_err[0][0]

                z_m2 = z2(xm2, ym2)  # Z of mirror in M2 coordinates

                root = zm2 - (z_m2 + z_err)
                return root

            t_m2 = optimize.brentq(root_z2, focal + 1000, focal + 12000)

            # Location of where ray hits M2
            x_m2 = x_0 + alpha * t_m2
            y_m2 = y_0 + beta * t_m2
            z_m2 = z_0 + gamma * t_m2

            # Using x and y in M2 coordiantes, find the z err:

            P_m2 = np.array([x_m2, y_m2, z_m2])

            ###### in M2 coordinates ##########################
            x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(
                x_m2, y_m2, z_m2
            )  # P_m2 temp
            x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(x_0, y_0, z_0)  # P_rx temp
            norm = d_z2(x_m2_temp, y_m2_temp)
            norm_temp = np.array([-norm[0], -norm[1], 1])
            N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
            vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
                [x_rx_temp, y_rx_temp, z_rx_temp]
            )
            dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
            tan_rx_m2 = vec_rx_m2 / dist_rx_m2

            # Outgoing ray
            tan_og = tan_rx_m2 - 2 * np.dot(np.sum(np.dot(tan_rx_m2, N_hat)), N_hat)

            # Transform back to telescope cordinates
            N_hat_t = np.zeros(3)

            N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
            N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
            N_hat_z_temp = N_hat[2]

            N_hat_t[0] = N_hat_x_temp
            N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
            N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)

            tan_rx_m2_t = np.zeros(3)

            tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(
                np.pi
            )
            tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(
                np.pi
            )
            tan_rx_m2_z_temp = tan_rx_m2[2]

            tan_rx_m2_t[0] = tan_rx_m2_x_temp
            tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(th2) - tan_rx_m2_z_temp * np.sin(
                th2
            )
            tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(th2) + tan_rx_m2_z_temp * np.cos(
                th2
            )

            tan_og_t = np.zeros(3)
            tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
            tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
            tan_og_z_temp = tan_og[2]

            tan_og_t[0] = tan_og_x_temp
            tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
            tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)
            ##################################################

            alpha = tan_og_t[0]
            beta = tan_og_t[1]
            gamma = tan_og_t[2]

            i_row = row_panm_m1[ii]
            i_col = col_panm_m1[ii]
            i_panm = np.where(
                (panel_model1[0, :] == i_row) & (panel_model1[1, :] == i_col)
            )
            if len(i_panm[0]) != 0:
                a = panel_model1[2, i_panm]
                b = panel_model1[3, i_panm]
                c = panel_model1[4, i_panm]
                d = panel_model1[5, i_panm]
                e = panel_model1[6, i_panm]
                f = panel_model1[7, i_panm]
                x0 = panel_model1[8, i_panm]
                y0 = panel_model1[9, i_panm]

                def root_z1(t):
                    x = P_m2[0] + alpha * t
                    y = P_m2[1] + beta * t
                    z = P_m2[2] + gamma * t
                    xm1, ym1, zm1 = tele_into_m1(
                        x, y, z
                    )  # take ray end coordinates and convert to M1 coordinates

                    xm1_err, ym1_err, zm1_err = tele_into_m1(x, y, z)

                    x_temp = xm1_err * np.cos(np.pi) + zm1_err * np.sin(np.pi)
                    y_temp = ym1_err
                    z_temp = -xm1_err * np.sin(np.pi) + zm1_err * np.cos(np.pi)

                    xpc = x_temp - x0
                    ypc = y_temp - y0

                    z_err = (
                        a
                        + b * xpc
                        + c * (ypc)
                        + d * (xpc ** 2 + ypc ** 2)
                        + e * (xpc * ypc)
                    )

                    z_err = z_err[0][0]
                    z_m1 = z1(xm1, ym1)  # Z of mirror 1 in M1 coordinates
                    root = zm1 - (z_m1 + z_err)
                    return root

                t_m1 = optimize.brentq(root_z1, 500, 22000)

                # Location of where ray hits M1
                x_m1 = P_m2[0] + alpha * t_m1
                y_m1 = P_m2[1] + beta * t_m1
                z_m1 = P_m2[2] + gamma * t_m1
                P_m1 = np.array([x_m1, y_m1, z_m1])

                ###### in M1 cordinates ##########################
                x_m1_temp, y_m1_temp, z_m1_temp = tele_into_m1(
                    x_m1, y_m1, z_m1
                )  # P_m2 temp
                x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m1(
                    P_m2[0], P_m2[1], P_m2[2]
                )  # P_rx temp
                norm = d_z1(x_m1_temp, y_m1_temp)
                norm_temp = np.array([-norm[0], -norm[1], 1])
                N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
                vec_m2_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array(
                    [x_m2_temp, y_m2_temp, z_m2_temp]
                )
                dist_m2_m1 = np.sqrt(np.sum(vec_m2_m1 ** 2))
                tan_m2_m1 = vec_m2_m1 / dist_m2_m1

                # Outgoing ray
                tan_og = tan_m2_m1 - 2 * np.dot(np.sum(np.dot(tan_m2_m1, N_hat)), N_hat)

                # Transform back to telescope cordinates
                N_hat_t = np.zeros(3)
                tan_m2_m1_t = np.zeros(3)
                tan_og_t = np.zeros(3)

                N_x_temp = N_hat[0] * np.cos(np.pi) + N_hat[2] * np.sin(np.pi)
                N_y_temp = N_hat[1]
                N_z_temp = -N_hat[0] * np.sin(np.pi) + N_hat[2] * np.cos(np.pi)
                N_hat_t[0] = N_x_temp
                N_hat_t[1] = N_y_temp * np.cos(tele_geo.th_1) - N_z_temp * np.sin(
                    tele_geo.th_1
                )
                N_hat_t[2] = N_y_temp * np.sin(tele_geo.th_1) + N_z_temp * np.cos(
                    tele_geo.th_1
                )

                tan_m2_m1_x_temp = tan_m2_m1[0] * np.cos(np.pi) + tan_m2_m1[2] * np.sin(
                    np.pi
                )
                tan_m2_m1_y_temp = tan_m2_m1[1]
                tan_m2_m1_z_temp = -tan_m2_m1[0] * np.sin(np.pi) + tan_m2_m1[
                    2
                ] * np.cos(np.pi)
                tan_m2_m1_t[0] = tan_m2_m1_x_temp
                tan_m2_m1_t[1] = tan_m2_m1_y_temp * np.cos(
                    tele_geo.th_1
                ) - tan_m2_m1_z_temp * np.sin(tele_geo.th_1)
                tan_m2_m1_t[2] = tan_m2_m1_y_temp * np.sin(
                    tele_geo.th_1
                ) + tan_m2_m1_z_temp * np.cos(tele_geo.th_1)

                tan_og_x_temp = tan_og[0] * np.cos(np.pi) + tan_og[2] * np.sin(np.pi)
                tan_og_y_temp = tan_og[1]
                tan_og_z_temp = -tan_og[0] * np.sin(np.pi) + tan_og[2] * np.cos(np.pi)
                tan_og_t[0] = tan_og_x_temp
                tan_og_t[1] = tan_og_y_temp * np.cos(
                    tele_geo.th_1
                ) - tan_og_z_temp * np.sin(tele_geo.th_1)
                tan_og_t[2] = tan_og_y_temp * np.sin(
                    tele_geo.th_1
                ) + tan_og_z_temp * np.cos(tele_geo.th_1)

                ##################################################

                dist_m1_ap = abs((z_ap - P_m1[2]) / tan_og_t[2])
                total_path_length = t_m2 + t_m1 + dist_m1_ap
                # total_path_length = dist_rx_m2 + dist_m2_m1 + dist_m1_ap
                pos_ap = P_m1 + dist_m1_ap * tan_og_t

                # # Estimate theta
                # de_ve = np.arctan(tan_rx_m2_t[2] / (-tan_rx_m2_t[1]))
                # de_ho = np.arctan(
                #     tan_rx_m2_t[0] / np.sqrt(tan_rx_m2_t[1] ** 2 + tan_rx_m2_t[2] ** 2)
                # )

                # power 
                power = np.exp(
                    (-0.5)
                    * ((th - theta_m) ** 2 + (ph - phi_m) ** 2)
                    / (horn_fwhp / (np.sqrt(8 * np.log(2)))) ** 2
                )
                # Write out

                return [
                        x_m2, y_m2, z_m2, 
                        x_m1, y_m1, z_m1, 
                        N_hat_t[0], N_hat_t[1], N_hat_t[2], 
                        pos_ap[0], pos_ap[1], pos_ap[2], 
                        tan_og_t[0], tan_og_t[1], tan_og_t[2], 
                        total_path_length, power,  # to determine phase, amp
                        ]
        return [0]*len(self.out)

    def output(self):

        if self.n_pts > 3000: # a threshold to use parallelism (it's a empirical number tested with Ryzen 5 3600)
            pool = mp.Pool(max(1, mp.cpu_count()-2))
            # self.out = np.array(list(pool.imap(self.trace_rays, range(self.n_pts)))).transpose()
            self.out = np.array(pool.map(self.trace_rays, range(self.n_pts))).transpose()
            pool.close()
            pool.join()
        else:
            result = []
            for ii in range(self.n_pts):
                result.append(self.trace_rays(ii))
            self.out = np.array(result).transpose()

        return self.out



def ray_mirror_pts(P_rx, tele_geo, theta, phi):
    rmp = RayMirrorPts(P_rx, tele_geo, theta, phi)
    return rmp.output()


def aperature_fields_from_panel_model(panel_model1, panel_model2, \
    P_rx, tele_geo, theta, phi, rxmirror
    ):
    affpm = AperatureFieldsFromPanelModel(panel_model1, panel_model2, \
        P_rx, tele_geo, theta, phi, rxmirror)
    return affpm.output()
 
if __name__ == "__main__":
    '''
    To test the code 
    '''
    
    import time
    import tele_geo as tg
    import ap_field as af
    import pan_mod as pm

    print("Calculating the aperture field.....")
    tele_geo_t = tg.initialize_telescope_geometry()

    # Panel Models with surface errors
    adj_1_A = np.random.randn(1092) * 20
    adj_2_A = np.random.randn(1092) * 20

    pan_mod2_tA = pm.panel_model_from_adjuster_offsets(
        2, adj_2_A, 1, 0
    )  #  on M2
    pan_mod1_tA = pm.panel_model_from_adjuster_offsets(
        1, adj_1_A, 1, 0
    ) 

    # FOV of RX (directions of outgoing rays from the receiver feed)
    N_th = 100
    N_ph = 100
    th = np.linspace(-np.pi / 2 - 0.28, -np.pi / 2 + 0.28, N_th)
    ph = np.linspace(np.pi / 2 - 0.28, np.pi / 2 + 0.28, N_ph)

    rx_t = np.array([0, 0, 0])
    # Path of the rays from the RX to the aperture plane
    print(f"Ray tracing WITH multiprocessing...") 
    time_start = time.time()
    rxmirror_t_mp = ray_mirror_pts(rx_t, tele_geo_t, th, ph)
    ap_field_mp = aperature_fields_from_panel_model(
        pan_mod1_tA, pan_mod2_tA, rx_t, tele_geo_t, th, ph, rxmirror_t_mp
    )
    print(f"Processing time : {time.time()-time_start}") 

    print(f"Ray tracing WITHOUT multiprocessing...") 
    time_start = time.time()
    rxmirror_t = af.ray_mirror_pts(rx_t, tele_geo_t, th, ph)
    ap_field = af.aperature_fields_from_panel_model(
        pan_mod1_tA, pan_mod2_tA, rx_t, tele_geo_t, th, ph, rxmirror_t
    )
    print(f"Processing time : {time.time()-time_start}") 
    
    assert rxmirror_t_mp.all() == rxmirror_t.all(), \
            "Inconsistency found. Please contact Tung at ctcheung@uchicago.edu for bug-fixing"
    assert ap_field_mp.all() == ap_field.all(), \
            "Inconsistency found. Please contact Tung at ctcheung@uchicago.edu for bug-fixing"