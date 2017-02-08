#!/bin/python
import numpy as num
import matplotlib.pyplot as plt
import copy
from os import path
from .abbreviations import _ageKey, _provinceKey, _referenceKey, _pubYear

THICKNESS_HALFSPACE = 2


class VelocityProfile:
    '''
    Single velocity profile representation from the Global Crustal Database

    https://earthquake.usgs.gov/data/crust/

    .. note ::

        Citation

        W.D. Mooney, G. Laske and G. Masters, CRUST 5.1: A global crustal model
        at 5°x5°. J. Geophys. Res., 103, 727-747, 1998.
    '''
    def __init__(self, lat, lon, Vp, Vs, d, h=None, **kwargs):
        '''Initialize the velocity profile

        :param lat: profiles latitude
        :type lat: float
        :param lon: profile longitude
        :type lon: float
        :param elevation: optional
        :type elevation: float
        :param Vp: P wave velocity as float list in km/s, if n/a .00
        :type Vp: :class:`numpy.array`
        :param Vs: P wave velocity as float list in km/s
        :type Vs: :class:`numpy.array`
        :param h: layer thickness
        :type h: :class:`numpy.array`
        :param geol_loc:
        :type geol_loc:
        :param geol_province: Geological province, abbreviation from
        :type geol_province: str
        :param geol_age: Geological age of the formation location
            as eon/epoch name
        :type geol_age: str
        :param method: Measurering method of the profile
        :type method: str
        :param heatflow: Heatflow measurement if available in [mW/m^2]
        :type heatflow: float
        :type reference: Reference/Citation for for profile
        :type reference: str
        :param uid: unique id from GSN list, default ``0``
        :type uid: int
        '''

        self.lat = lat
        self.lon = lon
        self.vp = num.array(Vp)
        self.vs = num.array(Vs)
        self.d = num.array(d)

        self.h = num.abs(self.d - num.roll(self.d, -1))
        self.h[-1] = 0

        self.nlayers = self.vp.size - 1

        self.geog_loc = kwargs.get('geog_loc', None)
        self.geol_province = kwargs.get('geol_province', None)
        self.geol_age = kwargs.get('age', None)
        self.elevation = kwargs.get('elevation', num.nan)
        self.method = kwargs.get('method', None)
        self.heatflow = kwargs.get('heatflow', num.nan)
        if self.heatflow == 0:
            self.heatflow = num.nan
        self.reference = kwargs.get('reference', None)
        self.pub_year = _pubYear(self.reference)
        self.uid = num.int(kwargs.get('uid', 0))

        self.vs[self.vs == 0] = num.nan
        self.vp[self.vp == 0] = num.nan

        self._step_depth = num.repeat(self.d, 2)
        self._step_depth[0] = 0
        self._step_depth[-1] += THICKNESS_HALFSPACE

        self._step_vp = num.repeat(self.vp, 2)
        self._step_vs = num.repeat(self.vs, 2)

        self._cont_vp = None
        self._cont_hash = None

    def continuousProfile(self, depths):
        '''
        function veloc_at_depth returns a continuous velocity function over
        depth, mind ``THICKNESS_HALFSPACE``

        :param depth: numpy.array vector of depths
        :param phase: wave phase, P or S, type string

        :self cont_depths: save continues depth param for performance
        :self cont_veloc: save contiues veloc for performance
        :return: array of depth corresponding velocities, type numpy.array
        '''
        # Caching purpose
        if self._cont_vp is not None and self._cont_hash == depths.sum():
            return self._cont_vp

        velocity = self.vp

        cont_v = num.empty(len(depths))
        cont_v[:] = num.nan
        _last_i = 0
        for _ti, this_depth in enumerate(depths):
            for _i in xrange(_last_i, self.d.size-1):
                if this_depth >= self.d[_i] and this_depth < self.d[_i+1]:
                    cont_v[_ti] = velocity[_i]
                    break
                elif this_depth >= self.d[-1]\
                and this_depth < (self.d[-1] + THICKNESS_HALFSPACE):
                    cont_v[_ti] = velocity[-1]
                    break
                _last_i = _i

        self._cont_vp = cont_v
        self._cont_hash = depths.sum()
        return cont_v

    def plot(self, figure=None, plt_vs=False):
        ''' Plot velocity - depth function through matplotlib '''
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        ax.plot(self._step_vp, -self._step_depth, color='k', linestyle= '-', linewidth=1.2)
        if num.any(self.vs) and plt_vs:
            ax.plot(self._step_vs, -self._step_depth, color='k', linestyle='--', linewidth=1.2)

        if figure is None:
            ax.set_title('Crustal Velocity Structure at ' + str(self.lat) + " / " + str(self.lon))
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Depth (km)')
            ax.plot([], [], color='k', linestyle='-', label='V$_P$', linewidth=1.2)
            if plt_vs:
                ax.plot([], [], color='k', linestyle='--', label='V$_S$', linewidth=1.2)
            ax.legend(loc=1)
            ax.grid(color='.4')
            fig.show()

    def __str__(self, csv=False):
        if not csv:
            output = '{:<24} {:05d}\n'.format('Profile UID:', self.uid)
            try:
                output += '{:<24} {:.2f} N, {:.2f} E\n'.format('Geographic Coordinates:', self.lat, self.lon)
                output += '{:<24} {:<6}\t({:})\n'.format('Geographical Location:', self.geog_loc, _provinceKey(self.geog_loc))
                output += '{:<24} {:.2f} km\n'.format('ETOPO5 Elevation:', self.elevation)
                output += '{:<24} {:.2f} W/m2\n'.format('Heatflow:', self.heatflow)
                output += '{:<24} {:}\n'.format('Geological Province:', self.geol_province)
                output += '{:<24} {:<4}\t({:})\n'.format('Geological Age:', self.geol_age, _ageKey(self.geol_age))
                output += '{:<24} {:<6}\t({:})\n'.format('Reference:', self.reference, _referenceKey(self.reference))
                output += '-'*30 + '\n'
            except:
                pass
            output += '%5s\t%6s\t%6s\t%6s\n' % ('Vp', 'Vs', 'H', 'Depth')
            for l in xrange(len(self.h)):
                output += '%5.2f\t%6.2f\t%6.2f\t%6.2f\n' % (self.vp[l], self.vs[l], self.h[l], self.d[l])
            return output
        elif csv:
            output = ''
            for _l in xrange(len(self.h)):
                # uid, Lat, Lon, Vp, Vs, H, Depth, Reference
                output += '%d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %s\n' % (
                    self.uid, self.lat, self.lon,
                    self.vp[_l], self.vs[_l], self.h[_l], self.d[_l],
                    self.reference
                    )
            return output

    def str_export(self):
        output = ''
        for _l in xrange(len(self.h)):
            output += '%.2f\t%.2f\t%.2f\n' % (self.h[_l], self.vs[_l], self.vp[_l])
        return output


class CrustDB(object):
    '''
    CrustDB extents class velocityProfile as it gathers velocity profiles
    and provides functions for spatial selection, querying and processing of the data.
    '''
    def __init__(self, database_file=None):
        self.profiles = []

        '''
        init list for selection lats and lons, velocities, thicknesses and depths
        '''
        self.data_matrix = None
        self.name = None

        if database_file:
            self._read(database_file)
        else:
            self._read(path.join(path.dirname(__file__),
                       'data/gsc20130501.txt'))

    def __len__(self):
        return len(self.profiles)

    def __setitem__(self, key, value):
        if not isinstance(value, VelocityProfile):
            raise TypeError('Element is not a VelocityProfile')
        self.profiles[key] = value

    def __delitem__(self, key):
        self.profiles.remove(key)

    def __getitem__(self, key):
        return self.profiles[key]

    def __str__(self):
        rstr = "Container contains " + str(len(self.profiles)) + " velocity profiles:\n\n"
        rstr += "uid\tlat\tlon\tmax depth\n"
        #for profile in self.profiles:
        #    rstr += str(profile.uid) + '\t'+ str(profile.lat) + '\t'+ str(profile.lon) + '\t'+ str(max(profile.depth)) + '\n'
        return rstr

    def append(self, value):
        if not isinstance(value, VelocityProfile):
            raise TypeError('Element is not a VelocityProfile')
        self.profiles.append(value)

    def copy(self):
        return copy.deepcopy(self)

    def lats(self):
        return num.array(
            [self.profiles[i].lat for i in xrange(len(self.profiles))])
    
    def lons(self):
        return num.array(
            [self.profiles[i].lon for i in xrange(len(self.profiles))])

    def dataMatrix(self):
        '''
        function init_data initializes continuous arrays of 4xNprofiles
        '''

        if self.data_matrix is not None:
            return self.data_matrix

        cont_vp = num.concatenate(
            [profile.vp for profile in self.profiles])
        cont_vs = num.concatenate(
            [profile.vs for profile in self.profiles])
        cont_h = num.concatenate(
            [profile.h for profile in self.profiles])
        cont_d = num.concatenate(
            [profile.d for profile in self.profiles])

        self.data_matrix = num.core.records.fromarrays(
            num.vstack([
                cont_vp,
                cont_vs,
                cont_h,
                cont_d]),
            names='vp, vs, h, d')
        return self.data_matrix

    def vMatrix(self, drange=(0, 60), dd=.1):
        '''
        function creates regular sampled velocity matrix

        :param drange: Min/Max Tuple of depth range to examine
        :param dd: Stepping in depth
        '''
        sdepth = num.linspace(drange[0], drange[1], (drange[1]-drange[0])/dd)
        ndepth = len(sdepth)

        v_mat = num.empty((len(self.profiles), ndepth))
        # Arrange data in 2-D array
        for _i, profile in enumerate(self.profiles):
            v_mat[_i, :] = profile.continuousProfile(sdepth)

        return sdepth, num.ma.masked_invalid(v_mat)

    def rmsRank(self, corr_profile, drange=(0, 35), dd=.1):
        '''
        Function correlates :param corr_profile: to each profile in the database

        :param drange: Tuple for correlation interval
        :param dd: Sample interval

        :returns rms: RMS ector length of N_profiles
        '''
        if not isinstance(corr_profile, velocityProfile):
            raise ValueError('corr_profile must be a crustshot.velocityProfile()')

        sdepth, p_matrix = self.vMatrix(drange, dd)
        v_corr = corr_profile.continuousProfile(sdepth)

        #filling nans with max Vp
        nan_ind = num.where(num.isnan(v_corr))[0]
        v_corr[nan_ind] = v_corr.max()

        rms = num.empty(len(self.profiles))
        for _p in xrange(len(self.profiles)):
            _corr_profile = p_matrix[_p,:]
            nan_ind = num.where(num.isnan(_corr_profile))[0]
            _corr_profile[nan_ind] = _corr_profile.max()

            rms[_p] = num.sqrt(((_corr_profile-v_corr)**2).sum()/len(v_corr))

        return rms

    def histogram2d(self, drange=(0, 60), vrange=(5.5, 8.5),
                    dd=.1, vbin=.1, dbin=2):
        '''
        Creates a 2D Histogram of all the velocity profiles
        check numpy.histogram2d for more information

        :param drange: Min/Max Tuple of depth range to examine
        :param dd: Stepping in depth

        :return: num.histogram2d
        '''
        # Arrange data in 1-D array
        sdepth, v_vec = self.vMatrix(drange, dd)
        v_vec = v_vec.flatten()
        d_vec = num.tile(sdepth, len(self.profiles))

        # Velocity and depth bins
        vbins = (vrange[1]-vrange[0]) / vbin
        dbins = int((drange[1]-drange[0]) // dbin)

        return num.histogram2d(v_vec, d_vec,
                            range=(vrange, drange),
                            bins=(vbins, dbins),
                            normed=False)


    def statsMean(self, drange=(0, 60), dd=.1):
        '''
        Returns the mean velocity profile plus std variation

        :param drange: Min/Max Tuple of depth range to examine
        :param dd: Stepping in depth

        :return sdepth: Regular spaced depth vector
        :return v_mean: Mean velocity at depth
        :return v_std: std for each velocity
        '''
        sdepth, v_mat = self.vMatrix(drange, dd)
        v_mean = num.ma.mean(v_mat, axis=0)
        v_std = num.ma.std(v_mat, axis=0)

        return sdepth, v_mean.flatten(), v_std.flatten()

    def statsMode(self, drange=(0, 60), dd=.1):
        '''
        Returns the mode velocity profile plus std variation

        :param drange: Min/Max Tuple of depth range to examine
        :param dd: Stepping in depth

        :return sdepth: Regular spaced depth vector
        :return v_mode: Mode velocity for each layer
        :return v_counts: Number of counts for each bin
        '''
        import scipy.stats

        sdepth, v_mat = self.vMatrix(drange, dd)
        v_mode, v_counts = scipy.stats.mstats.mode(v_mat, axis=0)
        return sdepth, v_mode.flatten(), v_counts.flatten()

    def plotStats(self, drange=(0, 60), dd=.1, figure=None):
        '''
        Plots the MODE, MEAN and STD

        :param drange: Min/Max Tuple of depth range to examine
        :param dd: Stepping in depth
        '''
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        sdepth, v_mean, v_std = self.statsMean(drange, dd)
        sdepth, v_mode, v_counts = self.statsMode(drange, dd)

        ax.plot(v_mode, sdepth, color='k')
        ax.plot(v_mean, sdepth, linestyle='--', color='k', alpha=.3)

        _stp = len(sdepth) // 10
        ax.errorbar(v_mean.data[::_stp], sdepth[::_stp], xerr=v_std.data[::_stp], fmt=None, alpha=.5)

        ax.set_xlabel('$Vp$ (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.invert_yaxis()

        if self.name is not None:
            ax.set_title((ax.get_title() + ' for ' + self.name))

        if figure is None:
            fig.show()


    def plotProfiles(self, figure=None, plt_all=False, plt_vs=False):
        '''
        function plot_v plots velocity depth funtions for :param selection: or the whole dataset

        :param selection: list of selected profiles, type numpy.array or list()
        '''
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        # Plot profiles
        for _i, profile in enumerate(self.profiles):
            profile.plot(figure=fig, plt_vs=plt_vs)
            if _i > 50 and not plt_all:
                break


        ax.text(.05, .05, '%d Profiles' % len(self.profiles),
            transform=ax.transAxes, fontsize=10, va='bottom', ha='left', alpha=.7)
        ax.set_title('Crustal Velocity Structure')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.plot([], [], color='k', linestyle='-', label='V$_P$', linewidth=1.2)
        if plt_vs:
            ax.plot([], [], color='k', linestyle='--', label='V$_S$', linewidth=1.2)
        ax.legend(loc=1)
        ax.grid(color='.4')

        if self.name is not None:
            ax.set_title((ax.get_title() + ' for ' + self.name))

        if figure is None:
            fig.show()


    def plotHistogram(self, range=(5.5, 8.5), bins=6*5, key='vp', figure=None):
        '''
        function plot_hist plots a velocity histogram within :param range: and :param bins:

        :param range: frequency range list (min, max), type list
        :param bins: number of bins, type int
        '''
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        data = self.dataMatrix()[key]

        ax.hist(data, weights=self.data_matrix['h'], 
                range=range, bins=bins, 
                color='g', alpha=.5)
        ax.text(.95, .95, '%d Profiles' % len(self.profiles),
                transform=ax.transAxes, fontsize=10, va='top', ha='right', alpha=.7)
        ax.set_title('Distribution of %s' % key.title())
        ax.set_xlabel('%s (km/s)' % key.title())
        ax.set_ylabel('Cumulative Occurrence')
        ax.yaxis.grid(alpha=.4)

        if self.name is not None:
            ax.set_title((ax.get_title() + ' for ' + self.name))

        if figure is None:
            fig.show()

    def plotHistogram2d(self, drange=(0, 60), dd=.1, dbin=2, vbin=.1, vrange=(5.5, 8.5),
                        plt_mode=True, plt_mean=True, plt_cbar=False, plt_max=50,
                        return_mode=False, percent=False,
                        figure=None, aspect=.02):
        '''
        function creates a two dimensional histogram

        :param drange: Min/Max Tuple of depth range to examine
        :param dd: Stepping in depth
        :param vrange: Min/Max Tuple of velocity range to examine
        :plt_mode: Boolean wheather to plot the Mode
        :plt_mean: Boolean wheather to plot the Mean
        '''
        if not isinstance(figure, plt.Figure):
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()

        ax = fig.add_axes([.1, .1, .7, .825])
        cbax = fig.add_axes([.85, .1, .025, .825])

        ax.set_anchor('SE')
        cbax.set_anchor('SW')

        # Velocity and depth bins
        vbin_size = vbin # km/s
        dbin_size = dbin # km
        vbins = (vrange[1]-vrange[0]) / vbin_size
        dbins = int((drange[1]-drange[0]) // dbin_size)

        vfield, xedg, yedg = self.histogram2d(
                            vrange=vrange, drange=drange,
                            vbin=vbin, dbin=dbin)

        vfield /= (dbin/dd)
        if percent:
            for d in xrange(vfield.shape[1]):
                vfield[:,d] /= vfield[:,d].sum()

        grid_ext = [xedg[0], xedg[-1], yedg[-1], yedg[0]]

        histogram = ax.imshow(vfield.swapaxes(0, 1),
                    interpolation='nearest',
                    extent=grid_ext, aspect=aspect)

        if plt_cbar:
            cticks = num.unique(num.arange(0, vfield.max(), vfield.max()//10).round())
            cbar = fig.colorbar(histogram, cax=cbax, ax=None, ticks=cticks, format='%1i')
            if percent:
                cbar.set_label('Percent')
            else:
                cbar.set_label('Number of Profiles')

        # Plot Mode and Mean if wished
        data_depth = num.linspace(drange[0], drange[1], dbins)
        data_mode = num.empty(len(data_depth))
        for _i in xrange(dbins):
            data_mode[_i] = vrange[0] + vfield[:, _i].argmax()*vbin_size
        data_mode += vbin_size/2

        if plt_mode:
            ax.plot(data_mode[data_depth<plt_max], data_depth[data_depth<plt_max], alpha=.8, color='w', label='Mode')

        if plt_mean:
            data_depth = num.linspace(drange[0], drange[1], dbins)
            data_mean = num.zeros_like(data_depth)
            for _d in xrange(dbins):
                cum_v = sum(vfield[:, _d])
                for _v in xrange(len(vfield[:, _d])):
                    data_mean[_d] += (vrange[0] + _v*vbin_size)*(vfield[_v, _d]/cum_v)
            data_mean += vbin_size/2

            ax.plot(data_mean[data_depth<plt_max], data_depth[data_depth<plt_max], alpha=.8, color='w', linestyle='--', label='Mean')

            # std_depth, std_mean, data_std = self.statsMean(dd=dbin_size, drange=drange)
            # _stp = len(std_depth) // 10
            # ax.errorbar(std_mean[::_stp], std_depth[::_stp], xerr=data_std[::_stp], fmt=None, alpha=.5, ecolor='w')

        # Plot labels and stuff
        ax.xaxis.set_ticks(num.arange(vrange[0], vrange[1], .5) + vbin_size/2)
        ax.xaxis.set_ticklabels(num.arange(vrange[0], vrange[1], .5))
        ax.grid(True, which="both", color="w", linewidth=.8, alpha=.4)

        ax.text(.025, .025, '%d Profiles' % len(self.profiles), color='w', alpha=.7,
            transform=ax.transAxes, fontsize=9, va='bottom', ha='left')

        ax.set_title('Crustal Velocity Distribution')
        ax.set_xlabel('V$_P$ (km/s)')
        ax.set_ylabel('Depth (km)')

        ax.set_xlim(vrange)

        if self.name is not None:
            ax.set_title((ax.get_title() + ' for ' + self.name))

        if plt_mode or plt_mean:
            leg = ax.legend(loc=1, fancybox=True, fontsize=10)
            leg.get_frame().set_alpha(.6)

        #clbar = plt.colorbar(histogram, shrink=.7)
        #clbar.set_label('Cummulative Occurance')
        if figure is None:
            plt.show()

        if return_mode:
            v_profile, v_index = num.unique(data_mode, return_index=True)
            return data_depth[v_index], v_profile-vbin_size/2

    def plotVelocitySurf(self, v_max, d_min=0, d_max=60, figure=None):
        '''
        Function triangulates a depth surface at velocity :v_max:

        :param v_max: maximal velocity, type float
        :param dz: depth is sampled in dz steps, type float
        :param d_max: maximum depth, type int
        :param d_min: minimum depth, type int
        :param phase: phase to query for, type string NOT YET IMPLEMENTED!!!
        :param figure: Plot into an existing matplotlib.figure
        '''
        m = self._basemap(figure)

        d = self.exceedVelocity(v_max, d_min, d_max)
        lons = self.lons()[d>0]
        lats = self.lats()[d>0]
        d = d[d>0]

        m.pcolor(lons, lats, d, latlon=True, tri=True,
            shading='faceted', alpha=1)
        m.colorbar()
        return self._basemapFinish(m, figure)

    def plotMap(self, figure=None, **kwargs):
        '''
        Function plots the current selected profiles

        :param figure: Plot into an existing matplotlib.figure
        :param **kwargs: Are passed on to plt.scatter() overwriting the
            default values
        '''
        m = self._basemap(figure)
        m.shadedrelief()
        # draw stations
        scatter_prop = {
            'color': 'r',
            's': 2.5,
            'label': '%s Selected Profiles' % len(self.profiles),
            'alpha': .7
        }
        scatter_prop.update(kwargs)
        st_x, st_y = m(self.lons(), self.lats())
        m.scatter(st_x, st_y, **scatter_prop)
        plt.gca().set_title('Geographical Locations of Velocity Profiles')
        plt.gca().text(.025, .025, '%d Profiles' % len(self.profiles),
            color='w', alpha=.5,
            transform=plt.gca().transAxes, fontsize=9, va='bottom', ha='left')

        # Draw legend
        leg = plt.gca().legend(loc=3, fancybox=True)
        leg.get_frame().set_alpha(.7)

        if self.name is not None:
            plt.gca().set_title((plt.gca().get_title() + ' for ' + self.name))

        return self._basemapFinish(m, figure)

    def _basemap(self, figure=None):
        '''
        Helper function returns empty matplotlib.basemap.

        :param figure: Plot into an existing matplotlib.figure
        '''
        from mpl_toolkits.basemap import Basemap

        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        lats = self.lats()
        lons = self.lons()
        frame_lon = num.abs(lons.max()-lons.min())*.075
        frame_lat = num.abs(lats.max()-lats.min())*.075

        corners = dict()
        corners['llcrnrlon'] = lons.min()-frame_lon
        if corners['llcrnrlon'] < -180: corners['llcrnrlon'] = -180
        corners['llcrnrlat'] = lats.min()-frame_lat
        if corners['llcrnrlat'] < -80: corners['llcrnrlat'] = -80

        corners['urcrnrlon'] = lons.max()+frame_lon
        if corners['urcrnrlon'] > 180: corners['urcrnrlon'] = 180
        corners['urcrnrlat'] = lats.max()+frame_lat
        if corners['urcrnrlat'] > 80: corners['urcrnrlat'] = 80

        map = Basemap(resolution='i', area_thresh=10000,
                    projection='merc',
                    lon_0=lons.min()+(lons.max()-lons.min())/2,
                    lat_0=lats.min()+(lats.max()-lats.min())/2,
                    ax=ax, **corners)
        return map

    def _basemapFinish(self, map, figure=None):
        '''
        Helper function finishes up basemap.

        :param figure: Plot into an existing matplotlib.figure
        '''
        map.drawcoastlines()
        map.drawcountries()
        map.drawstates()
        map.drawmeridians(num.arange(0,360,10), labels=[1,0,0,1], linewidth=.5)
        map.drawparallels(num.arange(-90,90,10), labels=[0,1,0,1], linewidth=.5)
        if figure is None:
            return plt.show()

    def exceedVelocity(self, v_max, d_min=0, d_max=60):
        '''
        function depth_at_veloc returns the last depth :param v_max: has not been exceeded.

        :param v_max: maximal velocity, type float
        :param dz: depth is sampled in dz steps, type float
        :param d_max: maximum depth, type int
        :param d_min: minimum depth, type int
        :param phase: phase to query for, type string NOT YET IMPLEMENTED!!!

        :return: list of lat, lon, depth and uid when v_max is exceeded, type list(num.array)
        '''
        self.profile_exceed_velocity = num.empty(len(self.profiles))
        self.profile_exceed_velocity[:] = num.nan

        for _p, profile in enumerate(self.profiles):
            for _i in xrange(len(profile.d)):
                if profile.d[_i] <= d_min\
                or profile.d[_i] >= d_max:
                    continue
                if profile.vp[_i] < v_max:
                    continue
                else:
                    self.profile_exceed_velocity[_p] = profile.d[_i]
                    break
        return self.profile_exceed_velocity

    ## Selection Functions
    def selectRegion(self, west, east, south, north):
        '''
        function select_region selects a region by geographic coordinates :param west, east, south, north:

        :param west: west edge of region, type float
        :param east: east edge of region, type float
        :param south: south edge of region, type float
        :param north: north edge of region, type float

        :return, self selection: array containing all profiles keys within desired region, type numpy.array
        '''
        # Select Region by lat and lon
        #

        r_container = self._emptyCopy()

        for profile in self.profiles:
            if profile.lon >= west and profile.lon <= east \
            and profile.lat <= north and profile.lat >= south:
                r_container.append(profile)

        return r_container

    def selectPolygon(self, poly):
        '''
        function select_poly determines if a profile is inside a given polygon or not
        polygon is a list of (x,y) pairs

        The algorithm is called the "Ray Casting Method"

        :param poly: list of x, y pairs, type list(numpy.array(2))

        :return, self selection: array containing all profiles keys within desired region, type numpy.array
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
            x = profile.lon
            y = profile.lat

            inside = False
            p1x,p1y = poly[0]
            for p2x, p2y in poly:
                if y >= min(p1y,p2y):
                    if y <= max(p1y,p2y):
                        if x <= max(p1x,p2x):
                            if p1y != p2y:
                                xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x,p1y = p2x,p2y
            if inside:
                r_container.append(profile)

        return r_container

    def selectLocation(self, lat, lon, radius=10):
        '''
        function select_location selects profiles at :param lat, lon: within a :param radius:

        :param lat: latitude, type float
        :param lon: longitude, type float
        :param radius: radius surrounding lat, lon, type float

        :return, self selection: array containing all profiles within radius of location lat, lon, type numpy.array
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
            if num.sqrt((lat - profile.lat)**2 + (lon - profile.lon)**2) <= radius:
                r_container.append(profile)

        return r_container

    def selectMinLayers(self, layers):
        '''
        selects profiles with more than :param layers:

        :param layers: Minimum number of layers, type int

        :return, self selection: array containing all profiles containing more than :param layers:
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
                if profile.layers >= layers:
                    r_container.append(profile)

        return r_container

    def selectMaxLayers(self, layers):
        '''
        selects profiles with more than :param layers:

        :param layers: Maximum number of layers, type int

        :return, self selection: array containing all profiles containing more than :param layers:
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
                if profile.layers <= layers:
                    r_container.append(profile)

        return r_container


    def selectMinDepth(self, depth):
        '''
        function select_depth selects only that profiles deeper than :param depth:
        either for the whole dataset or, if set, for the selected datasets

        :param depth: minimum depth for the profiles, type float

        :return, self selection: array containing all profiles keys deeper than desired depth, type numpy.array
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
            if profile.d.max() >= depth:
                r_container.append(profile)
        return r_container

    def selectMaxDepth(self, depth):
        '''
        function select_depth selects only that profiles shallower than :param depth:
        either for the whole dataset or, if set, for the selected datasets

        :param depth: minimum depth for the profiles, type float

        :return, self selection: array containing all profiles keys deeper than desired depth, type numpy.array
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
            if profile.d.max() <= depth:
                r_container.append(profile)
        return r_container

    def selectVp(self):
        r_container = self._emptyCopy()

        for profile in self.profiles:
            if not num.all(num.isnan(profile.vp)):
                r_container.append(profile)
        return r_container

    def selectVs(self):
        r_container = self._emptyCopy()

        for profile in self.profiles:
            if not num.all(num.isnan(profile.vs)):
                r_container.append(profile)
        return r_container

    def _emptyCopy(self):
        '''
        Helper function to return an empty container with the same
        base properties
        '''

        r_container = containerProfiles()
        r_container.name = self.name
        return r_container


    def writeCSV(self, filename=None):
        '''
        Function writes a CSV file as specified in the header below

        :param filename: File to write to.
        '''
        if filename is None:
            msg = 'Define a filename for CSV output'
            raise ExceptionError(msg)

        file = open(filename, 'w')
        header = '# uid, Lat, Lon, Vp, Vs, H, Depth, Reference\n'
        file.write(header)

        for profile in self.profiles:
            file.write(profile.__str__(csv=True).replace('nan', '0.00'))
        file.close()

    def writeReadable(self, filename=None):
        '''
        Function writes a readable file to :filename:

        :param filename: File to write to.
        '''
        if filename is None:
            msg = 'Define a filename for Readable output'
            raise ExceptionError(msg)

        file = open(filename, 'w')
        for profile in self.profiles:
            file.write(profile.__str__() + '\n')
        file.close()

    @classmethod
    def readDatabase(cls, database_file):
        db = cls()
        CrustDB._read(db, database_file)
        return db

    def _read(self, database_file):
        '''
        function readProfiles reads in the the GSN databasefile and puts it in containerProfiles

        !!TO BE INCLUDED INTO containerProfiles!!

        File format:

           uid  lat/lon  vp    vs    hc     depth
            2   29.76N   2.30   .00   2.00    .00  s  25.70   .10    .00  NAC-CO   5 U
                96.31W   3.94   .00   5.30   2.00  s  33.00   MCz  39.00  61C.3    EXC
                         5.38   .00  12.50   7.30  c
                         6.92   .00  13.20  19.80  c
                         8.18   .00    .00  33.00  m

            3   34.35N   3.00   .00   3.00    .00  s  35.00  1.60    .00  NAC-BR   4 R
               117.83W   6.30   .00  16.50   3.00     38.00   MCz  55.00  63R.1    ORO
                         7.00   .00  18.50  19.50
                         7.80   .00    .00  38.00  m


        :param database_file: path to database file, type string

        :return: class containerProfiles() containing all entries from database, type containerProfiles()
        '''

        with open(database_file, 'r') as database:
            vp = []; vs = []; h = []; depth = [];
            profile_info = {
                'uid': num.nan,
                'geol_province': None,
                'geog_loc': None,
                'elevation': num.nan,
                'heatflow': num.nan,
                'age': None,
                'method': None,
                'reference': None
            }
            readline = 1
            db = database.readlines()
            for dbline in db:
                if dbline.isspace():
                    # this is a entry!
                    # create velocity profile and append to container
                    if not len(depth) == 0:
                        self.append(VelocityProfile(
                            Vp=vp, Vs=vs, h=h, d=depth,
                            lat=lat, lon=lon,
                            **profile_info))
                    # for debugging purpose, if the databasefile is corrupt
                    if not len(vp) == len(h):
                        print lat, lon, vp, vs, h, depth, profile_info
                    # reset variables
                    vp = []; vs = []; h = []; depth = [];
                    readline = 1
                else:
                    # read in profile data
                    if readline == 1:
                        lat = float(dbline[8:13])
                        if dbline[13] == "S": lat = -lat
                        # Additional data
                        profile_info['uid'] = int(dbline[0:6])
                        profile_info['elevation'] = float(dbline[52:57])
                        profile_info['heatflow'] = float(dbline[58:64])
                        profile_info['geog_loc'] = dbline[66:72].strip()
                        profile_info['method'] = dbline[77]
                    if readline == 2:
                        lon = float(dbline[7:13])
                        if dbline[13] == "W":
                            lon = -lon
                        # Additional data
                        profile_info['age'] = dbline[54:58].strip()
                        profile_info['reference'] = dbline[66:72].strip()
                        profile_info['geol_province'] = dbline[74:78].strip()

                    try:
                        vp.append(float(dbline[17:21]))
                        vs.append(float(dbline[23:27]))
                        h.append(float(dbline[28:34]))
                        depth.append(float(dbline[35:41]))
                    except:
                        pass

                    readline += 1
            # Append last profile
            self.append(VelocityProfile(
                Vp=vp, Vs=vs, h=h, d=depth,
                lat=lat, lon=lon,
                **profile_info))

'''
Database is read in and crustal depth through velocities are queried
'''
# us = {
#     'west': -130,
#     'east': -60,
#     'south': 20,
#     'north': 55
# }
# test = readDatabase('gscnew.txt')
# us = test.selectRegion(**us)
