#!/bin/python
import numpy as num
import matplotlib.pyplot as plt
import copy
from os import path
from .abbreviations import ageKey, provinceKey, referenceKey, pubYear

THICKNESS_HALFSPACE = 2


class DatabaseError(Exception):
    pass


class ProfileEmpty(Exception):
    pass


class VelocityProfile:
    '''
    Single velocity profile representation from the Global Crustal Database

    https://earthquake.usgs.gov/data/crust/

    .. note ::

        Citation

        W.D. Mooney, G. Laske and G. Masters, CRUST 5.1: A global crustal model
        at 5°x5°. J. Geophys. Res., 103, 727-747, 1998.
    '''

    def __init__(self, lat, lon, vp, vs, d, h=None, **kwargs):
        '''Initialize the velocity profile

        :param lat: profiles latitude
        :type lat: float
        :param lon: profile longitude
        :type lon: float
        :param elevation: Elevation
        :type elevation: float, optional
        :param vp: P wave velocity as float list in km/s, if n/a .00
        :type vp: :class:`numpy.ndarray`
        :param vs: P wave velocity as float list in km/s
        :type vs: :class:`numpy.ndarray`
        :param h: layer thickness
        :type h: :class:`numpy.ndarray`
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
        :param uid: unique id from GSN list, defaults to ``0``
        :type uid: int
        '''

        self.lat = lat
        self.lon = lon
        self.vp = num.array(vp)
        self.vs = num.array(vs)
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
        self.reference = kwargs.get('reference', None)
        self.pub_year = pubYear(self.reference)
        self.uid = num.int(kwargs.get('uid', 0))

        self.vs[self.vs == 0] = num.nan
        self.vp[self.vp == 0] = num.nan

        self._step_vp = num.repeat(self.vp, 2)
        self._step_vs = num.repeat(self.vs, 2)
        self._step_depth = num.roll(num.repeat(self.d, 2), -1)
        self._step_depth[-1] = self._step_depth[-2] + THICKNESS_HALFSPACE

        self._interp_profile = {}

    def interpolateProfile(self, depths, phase='p'):
        '''
        function veloc_at_depth returns a continuous velocity function over
        depth

        :param depth: numpy.ndarray vector of depths
        :type depth: :class:`numpy.ndarray`
        :param phase: P or S wave velocity, ``['p', 's']``
        :type phase: str, optional
        :returns: velocities at requested depth
        :rtype: :py:`numpy.ndarray`
        '''

        if phase not in ['s', 'p']:
            raise AttributeError('Phase has to be either \'p\' or \'s\'.')
        if phase == 'p':
            vel = self._step_vp
        elif phase == 's':
            vel = self._step_vs

        if vel.size == 0:
            raise ProfileEmpty('Phase %s does not contain velocities' % phase)

        try:
            res = num.interp(depths, self._step_depth, vel,
                             left=num.nan, right=num.nan)
        except ValueError:
            raise ValueError('Could not interpolate velocity profile.')

        return res

    def plot(self, figure=None, plt_vs=False):
        ''' Plot velocity - depth function through matplotlib '''
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        ax.plot(self._step_vp, -self._step_depth,
                color='k', linestyle='-', linewidth=1.2)
        if self.has_s and plt_vs:
            ax.plot(self._step_vs, -self._step_depth,
                    color='k', linestyle='--', linewidth=1.2)

        if figure is None:
            ax.set_title('Crustal Velocity Structure at {p.lat:.4f}N, '
                         ' {p.lat:.4f}E'.format(p=self))
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Depth (km)')
            ax.plot([], [],
                    color='k', linestyle='-',
                    label='V$_P$', linewidth=1.2)
            if plt_vs:
                ax.plot([], [],
                        color='k', linestyle='--',
                        label='V$_S$', linewidth=1.2)
            ax.legend(loc=1)
            ax.grid(color='.4')
            fig.show()

    @property
    def geog_loc_long(self):
        return provinceKey(self.geog_loc)

    @property
    def geol_age_long(self):
        return ageKey(self.geol_age)

    @property
    def has_s(self):
        return num.any(self.vp)

    @property
    def has_p(self):
        return num.any(self.vs)

    def _csv(self):
        output = ''
        for d in xrange(len(self.h)):
            # uid, Lat, Lon, vp, vs, H, Depth, Reference
            output += ('{p.uid}, {p.lat}, {p.lon},'
                       ' {vp}, {vs}, {h}, {d}, {self.reference}').format(
                p=self,
                vp=self.vs[d], vs=self.vp[d], h=self.h[d], d=self.d[d])
        return output

    def __str__(self, csv=False):
        if csv:
            return self._csv()
        rstr = '''
Profile UID {:05d}
=================
Geographic Coordinates: {p.lat:.4f} N, {p.lon:.4f} N,
Geographical Location:  {p.geog_loc} ({p.geog_loc_long})
Number of layers:       {p.nlayers}
P Velocity Profile:     {p.has_p}
S Velocity Profile:     {p.has_s}
Measurement Method:     {p.method}
ETOPO5 Elevation:       {p.elevation}
Heatflow:               {p.heatflow:.4f} W/m2
Geological Province:    {p.geol_province}
Geological Age:         {p.geol_age} ({p.geol_age_long})
Reference:              {p.reference}
'''.format(p=self)
        return rstr


class CrustDB(object):
    '''
    CrustDB  is a container for VelocityProfiles and provides functions for
    spatial selection, querying, processing and visualising data.
    '''

    def __init__(self, database_file=None):
        self.profiles = []
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
        rstr = "Container contains %d velocity profiles:\n\n" % self.nprofiles
        return rstr

    @property
    def nprofiles(self):
        return len(self.profiles)

    def append(self, value):
        if not isinstance(value, VelocityProfile):
            raise TypeError('Element is not a VelocityProfile')
        self.profiles.append(value)

    def copy(self):
        return copy.deepcopy(self)

    def lats(self):
        return num.array(
            [p.lat for p in self.profiles])

    def lons(self):
        return num.array(
            [p.lon for p in self.profiles])

    def _dataMatrix(self):
        if self.data_matrix is not None:
            return self.data_matrix

        self.data_matrix = num.core.records.fromarrays(
            num.vstack([
                num.concatenate([p.vp for p in self.profiles]),
                num.concatenate([p.vs for p in self.profiles]),
                num.concatenate([p.h for p in self.profiles]),
                num.concatenate([p.d for p in self.profiles])
            ]),
            names='vp, vs, h, d')
        return self.data_matrix

    def velocityMatrix(self, drange=(0, 60), dd=.1, phase='p'):
        '''Create a regular sampled velocity matrix

        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param phase: Phase to calculate ``p`` or ``s``, defaults to ``p``
        :type phase: str

        :returns:
        :rtype: tuple, (sample_depth, :class:`numpy.ndarray`)
        '''
        dmin, dmax = drange
        sdepth = num.linspace(dmin, dmax, (dmax - dmin) / dd)
        ndepth = len(sdepth)

        vel_mat = num.empty((self.nprofiles, ndepth))
        for ip, profile in enumerate(self.profiles):
            vel_mat[ip, :] = profile.interpolateProfile(sdepth, phase=phase)

        return sdepth, num.ma.masked_invalid(vel_mat)

    def rmsRank(self, ref_profile, drange=(0, 35), dd=.1, phase='p'):
        '''Correlates ``ref_profile`` to each profile in the database

        :param ref_profile: Reference profile
        :type ref_profile: :class:`VelocityProfile`
        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple, optional
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param phase: Phase to calculate ``p`` or ``s``, defaults to ``p``
        :type phase: str

        :returns rms: RMS factor length of N_profiles
        :rtype: :class:`numpy.ndarray`
        '''
        if not isinstance(ref_profile, VelocityProfile):
            raise ValueError('ref_profile is not a VelocityProfile')

        sdepth, vel_matrix = self.velocityMatrix(drange, dd, phase=phase)
        ref_vel = ref_profile.interpolateProfile(sdepth, phase=phase)

        rms = num.empty(self.nprofiles)
        for p in xrange(self.nprofiles):
            profile = vel_matrix[p, :]
            rms[p] = num.sqrt(profile**2 - ref_vel**2).sum() / ref_vel.size
        return rms

    def histogram2d(self, drange=(0, 60), vrange=(5.5, 8.5),
                    dd=.1, dvbin=.1, ddbin=2, phase='p'):
        '''Create a 2D Histogram of all the velocity profiles

        Check :func:`numpy.histogram2d` for more information.

        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple
        :param vrange: Depth range, ``(vmin, vmax)``
        :type vrange: tuple
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param dvbin: Bin size in velocity dimension [km/s], defaults to .1
        :type dvbin: float
        :param dvbin: Bin size in depth dimension [km], defaults to 2
        :type dvbin: float
        :param phase: Phase to calculate ``p`` or ``s``, defaults to ``p``
        :type phase: str

        :return: 2D histogram
        :rtype: :class:`numpy.ndarray`
        '''
        sdepth, v_vec = self.velocityMatrix(drange, dd, phase=phase)
        v_vec = v_vec.flatten()
        d_vec = num.tile(sdepth, self.nprofiles)

        # Velocity and depth bins
        vbins = int((vrange[1] - vrange[0]) / dvbin)
        dbins = int((drange[1] - drange[0]) / ddbin)

        return num.histogram2d(v_vec, d_vec,
                               range=(vrange, drange),
                               bins=(vbins, dbins),
                               normed=False)

    def statsMean(self, drange=(0, 60), dd=.1, phase='p'):
        '''Mean velocity profile plus std variation

        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param phase: Phase to calculate ``p`` or ``s``, defaults to ``p``
        :type phase: str

        :returns: depth vector, mean velocities, standard deviations
        :rtype: tuple of :class:`numpy.ndarray`
        '''
        sdepth, v_mat = self.velocityMatrix(drange, dd, phase=phase)
        v_mean = num.ma.mean(v_mat, axis=0)
        v_std = num.ma.std(v_mat, axis=0)

        return sdepth, v_mean.flatten(), v_std.flatten()

    def statsMode(self, drange=(0, 60), dd=.1, phase='p'):
        '''Mode velocity profile plus std variation

        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param phase: Phase to calculate ``p`` or ``s``, defaults to ``p``
        :type phase: str

        :returns: depth vector, mode velocity, number of counts at each depth
        :rtype: tuple of :class:`numpy.ndarray`
        '''
        import scipy.stats

        sdepth, v_mat = self.velocityMatrix(drange, dd)
        v_mode, v_counts = scipy.stats.mstats.mode(v_mat, axis=0)
        return sdepth, v_mode.flatten(), v_counts.flatten()

    def statsMedian(self, drange=(0, 60), dd=.1, phase='p'):
        '''Median velocity profile plus std variation

        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param phase: Phase to calculate ``p`` or ``s``, defaults to ``p``
        :type phase: str

        :returns: depth vector, median velocities, standard deviations
        :rtype: tuple of :class:`numpy.ndarray`
        '''
        sdepth, v_mat = self.velocityMatrix(drange, dd, phase=phase)
        v_mean = num.ma.median(v_mat, axis=0)
        v_std = num.ma.std(v_mat, axis=0)

        return sdepth, v_mean.flatten(), v_std.flatten()

    def plotStats(self, drange=(0, 60), dd=.1, phase='p', figure=None):
        ''' Plots the MODE, MEAN and STD

        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param phase: Phase to plot ``p`` or ``s``, defaults to ``p``
        :type phase: str
        '''
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        sdepth, v_mean, v_std = self.statsMean(drange, dd, phase=phase)
        _, v_mode, v_counts = self.statsMode(drange, dd, phase=phase)
        _, v_median, _ = self.statsMedian(drange, dd, phase=phase)

        ax.plot(v_mode, sdepth,
                color='k', label='Mode')
        ax.plot(v_mean, sdepth,
                linestyle='--', color='k', alpha=.4, label='Mean')
        ax.plot(v_median, sdepth,
                linestyle='.', color='k', alpha=.4, label='Median')

        _stp = len(sdepth) // 10
        ax.errorbar(v_mean.data[::_stp], sdepth[::_stp],
                    xerr=v_std.data[::_stp], fmt=None, alpha=.5)

        ax.set_xlabel('$vp$ (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.legend(loc=1, alpha=.6)
        ax.invert_yaxis()

        if self.name is not None:
            ax.set_title('%s for %s' % (ax.get_title(), self.name))

        if figure is None:
            fig.show()

    def plotProfiles(self, figure=None, nplots=50, plt_vs=False):
        """Plot velocity depth funtions of the whole dataset

        :param figure: Figure to plot into, defaults to None
        :type figure: :class:`matplotlib.Figure, optional
        :param plt_all: Number of profiles to plot, defaults to 50
        :type plt_all: int, optional
        :param plt_vs: Plot S Wave velocities, defaults to False
        :type plt_vs: bool, optional
        """
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        for _i, profile in enumerate(self.profiles):
            profile.plot(figure=fig, plt_vs=plt_vs)
            if _i > nplots:
                break

        ax.set_title('Crustal Velocity Structure')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.text(.05, .05, '%d Profiles' % self.nprofiles,
                transform=ax.transAxes, fontsize=10,
                va='bottom', ha='left', alpha=.7)
        # Legend stuff
        ax.plot([], [],
                color='k', linestyle='-',
                label='V$_P$', linewidth=1.2)
        if plt_vs:
            ax.plot([], [],
                    color='k', linestyle='--',
                    label='V$_S$', linewidth=1.2)
        ax.legend(loc=1)
        ax.grid(alpha=.4)

        if self.name is not None:
            ax.set_title('%s for %s' % (ax.get_title(), self.name))

        if figure is None:
            fig.show()

    def plotHistogram(self, vrange=(5.5, 8.5), bins=6 * 5, key='vp',
                      figure=None):
        '''Plot 1D histogram of seismic velocities in the container

        :param vrange: Velocity range, defaults to (5.5, 8.5)
        :type vrange: tuple, optional
        :param bins: bins, defaults to 30 (see :func:`numpy.histogram`)
        :type bins: int, optional
        :param key: Property to plot out of ``['vp', 'vs', 'h', 'd']``,
            defaults to 'vp'
        :type key: str, optional
        :param figure: Figure to plot in, defaults to None
        :type figure: :class:`matplotlib.Figure`, optional
        '''
        if not isinstance(figure, plt.Figure):
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.gca()

        data = self._dataMatrix()[key]

        ax.hist(data, weights=self.data_matrix['h'],
                range=vrange, bins=bins,
                color='g', alpha=.5)
        ax.text(.95, .95, '%d Profiles' % len(self.profiles),
                transform=ax.transAxes, fontsize=10,
                va='top', ha='right', alpha=.7)

        ax.set_title('Distribution of %s' % key.title())
        ax.set_xlabel('%s (km/s)' % key.title())
        ax.set_ylabel('Cumulative Occurrence')
        ax.yaxis.grid(alpha=.4)

        if self.name is not None:
            ax.set_title('%s for %s' % (ax.get_title(), self.name))

        if figure is None:
            fig.show()

    def plotHistogram2d(self, drange=(0, 60), dd=.1, ddbin=2, dvbin=.1,
                        vrange=(5.5, 8.5), percent=False,
                        show_mode=True, show_mean=True, show_median=True,
                        show_cbar=False,
                        aspect=.02,
                        figure=None):
        ''' Plot a two 2D Histogram of seismic velocities

        :param drange: Depth range, ``(dmin, dmax)``, defaults to ``(0, 60)``
        :type drange: tuple
        :param vrange: Velocity range, ``(vmin, vmax)``
        :type vrange: tuple
        :param dd: Stepping in [km], defaults to ``.1``
        :type dd: float
        :param dvbin: Bin size in velocity dimension [km/s], defaults to .1
        :type dvbin: float
        :param dvbin: Bin size in depth dimension [km], defaults to 2
        :type dvbin: float
        :param phase: Phase to calculate ``p`` or ``s``, defaults to ``p``
        :type phase: str

        :param drange: Min/Max Tuple of depth range to examine
        :param dd: Stepping in depth
        :param vrange: Min/Max Tuple of velocity range to examine
        :show_mode: Boolean wheather to plot the Mode
        :show_mean: Boolean wheather to plot the Mean
        '''
        if not isinstance(figure, plt.Figure):
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()

        ax = fig.gca()

        vmin, vmax = vrange
        dmin, dmax = drange

        vfield, xedg, yedg = self.histogram2d(vrange=vrange, drange=drange,
                                              dd=dd, dvbin=dvbin, ddbin=ddbin)
        vfield /= (ddbin / dd)

        if percent:
            vfield /= vfield.sum(axis=1)[num.newaxis, :]

        grid_ext = [xedg[0], xedg[-1], yedg[-1], yedg[0]]
        histogram = ax.imshow(vfield.swapaxes(0, 1),
                              interpolation='nearest',
                              extent=grid_ext, aspect=aspect)

        if show_cbar:
            cticks = num.unique(
                num.arange(0, vfield.max(), vfield.max() // 10).round())
            cbar = fig.colorbar(histogram, ticks=cticks, format='%1i')
            if percent:
                cbar.set_label('Percent')
            else:
                cbar.set_label('Number of Profiles')

        if show_mode:
            sdepth, vel_mode, _ = self.statsMode(drange=drange, dd=dd)
            ax.plot(vel_mode[sdepth < dmax] + dd/2, sdepth[sdepth < dmax],
                    alpha=.8, color='w', label='Mode')

        if show_mean:
            sdepth, vel_mean, _ = self.statsMean(drange=drange, dd=dd)
            ax.plot(vel_mean[sdepth < dmax] + dd/2, sdepth[sdepth < dmax],
                    alpha=.8, color='w', linestyle='--', label='Mean')

        if show_median:
            sdepth, vel_median, _ = self.statsMedian(drange=drange, dd=dd)
            ax.plot(vel_median[sdepth < dmax] + dd/2, sdepth[sdepth < dmax],
                    alpha=.8, color='w', linestyle=':', label='Median')

        # Finish plot
        ax.xaxis.set_ticks(num.arange(vmin, vrange[1], .5) + dvbin / 2)
        ax.xaxis.set_ticklabels(num.arange(vmin, vrange[1], .5))
        ax.grid(True, which="both", color="w", linewidth=.8, alpha=.4)

        ax.text(.025, .025, '%d Profiles' % self.nprofiles,
                color='w', alpha=.7,
                transform=ax.transAxes, fontsize=9, va='bottom', ha='left')

        ax.set_title('Crustal Velocity Distribution')
        ax.set_xlabel('V$_P$ (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.set_xlim(vrange)

        if self.name is not None:
            ax.set_title('%s for %s' % (ax.get_title(), self.name))

        if show_mode or show_mean:
            leg = ax.legend(loc=1, fancybox=True, fontsize=10)
            leg.get_frame().set_alpha(.6)

        if figure is None:
            plt.show()

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
        lons = self.lons()[d > 0]
        lats = self.lats()[d > 0]
        d = d[d > 0]

        m.pcolor(lons, lats, d, latlon=True, tri=True,
                 shading='faceted', alpha=1)
        m.colorbar()
        return self._basemapFinish(m, figure)

    def plotMap(self, figure=None, **kwargs):
        '''
        Function plots the currently selected profiles

        :param figure: Plot into an existing matplotlib.figure
        :param **kwargs: Are passed on to plt.scatter() overwriting the
            defaults to values
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
        frame_lon = num.abs(lons.max() - lons.min()) * .075
        frame_lat = num.abs(lats.max() - lats.min()) * .075

        corners = dict()
        corners['llcrnrlon'] = lons.min() - frame_lon
        if corners['llcrnrlon'] < -180:
            corners['llcrnrlon'] = -180
        corners['llcrnrlat'] = lats.min() - frame_lat
        if corners['llcrnrlat'] < -80:
            corners['llcrnrlat'] = -80

        corners['urcrnrlon'] = lons.max() + frame_lon
        if corners['urcrnrlon'] > 180:
            corners['urcrnrlon'] = 180
        corners['urcrnrlat'] = lats.max() + frame_lat
        if corners['urcrnrlat'] > 80:
            corners['urcrnrlat'] = 80

        map = Basemap(resolution='i', area_thresh=10000,
                      projection='merc',
                      lon_0=lons.min() + (lons.max() - lons.min()) / 2,
                      lat_0=lats.min() + (lats.max() - lats.min()) / 2,
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
        map.drawmeridians(
            num.arange(
                0, 360, 10), labels=[
                1, 0, 0, 1], linewidth=.5)
        map.drawparallels(num.arange(-90, 90, 10),
                          labels=[0, 1, 0, 1], linewidth=.5)
        if figure is None:
            return plt.show()

    def exceedVelocity(self, v_max, d_min=0, d_max=60):
        ''' Returns the last depth ``v_max`` has not been exceeded.

        :param v_max: maximal velocity
        :type vmax: float
        :param dz: depth is sampled in dz steps
        :type dz: float
        :param d_max: maximum depth
        :type d_max: int
        :param d_min: minimum depth
        :type d_min: int

        :return: Lat, Lon, Depth and uid where ``v_max`` is exceeded
        :rtype: list(num.array)
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

    def selectRegion(self, west, east, south, north):
        '''
        function select_region selects a region by geographic coordinates

        :param west: west edge of region
        :type west: float
        :param east: east edge of region
        :type east: float
        :param south: south edge of region
        :type south: float
        :param north: north edge of region
        :type north: float

        :returns: All profile keys within desired region
        :rtype: :class:`numpy.ndarray`
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

        :param poly: list of x, y pairs, type list(numpy.ndarray(2))

        :return, self selection: array containing all profiles keys within desired region, type numpy.ndarray
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
            x = profile.lon
            y = profile.lat

            inside = False
            p1x, p1y = poly[0]
            for p2x, p2y in poly:
                if y >= min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / \
                                    (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                r_container.append(profile)

        return r_container

    def selectLocation(self, lat, lon, radius=10):
        '''
        function select_location selects profiles at :param lat, lon: within a :param radius:

        :param lat: latitude, type float
        :param lon: longitude, type float
        :param radius: radius surrounding lat, lon, type float

        :return, self selection: array containing all profiles within radius of location lat, lon, type numpy.ndarray
        '''
        r_container = self._emptyCopy()

        for profile in self.profiles:
            if num.sqrt((lat - profile.lat)**2 +
                        (lon - profile.lon)**2) <= radius:
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

        :return, self selection: array containing all profiles keys deeper than desired depth, type numpy.ndarray
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

        :return, self selection: array containing all profiles keys deeper than desired depth, type numpy.ndarray
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
        header = '# uid, Lat, Lon, vp, vs, H, Depth, Reference\n'
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

        '''

        def get_empty_record():
            meta = {
                'uid': num.nan,
                'geol_province': None,
                'geog_loc': None,
                'elevation': num.nan,
                'heatflow': num.nan,
                'age': None,
                'method': None,
                'reference': None
            }
            return [], [], [], [], meta

        vp, vs, h, depth, meta = get_empty_record()
        rec_line = 0
        with open(database_file, 'r') as database:
            for line, dbline in enumerate(database.readlines()):
                if dbline.isspace():
                    if not len(depth) == 0:
                        self.append(
                            VelocityProfile(vp=num.array(vp), vs=num.array(vs),
                                            h=num.array(h), d=num.array(depth),
                                            lat=lat, lon=lon,
                                            **meta))
                    # for debugging, if the database is corrupt
                    if not len(vp) == len(h):
                        raise DatabaseError(
                            'Inconsistent database, check line %d!\n\tDebug: '
                            % line, lat, lon, vp, vs, h, depth, meta)

                    vp, vs, h, depth, meta = get_empty_record()
                    rec_line = 0
                else:
                    try:
                        if rec_line == 0:
                            lat = float(dbline[8:13])
                            if dbline[13] == "S":
                                lat = -lat
                            # Additional meta data
                            meta['uid'] = int(dbline[0:6])
                            meta['elevation'] = float(dbline[52:57])
                            meta['heatflow'] = float(dbline[58:64])
                            meta['geog_loc'] = dbline[66:72].strip()
                            meta['method'] = dbline[77]
                        if rec_line == 1:
                            lon = float(dbline[7:13])
                            if dbline[13] == "W":
                                lon = -lon
                            # Additional meta data
                            meta['age'] = dbline[54:58].strip()
                            meta['reference'] = dbline[66:72].strip()
                            meta['geol_province'] = dbline[74:78].strip()
                        try:
                            vp.append(float(dbline[17:21]))
                            vs.append(float(dbline[23:27]))
                            h.append(float(dbline[28:34]))
                            depth.append(float(dbline[35:41]))
                        except ValueError:
                            pass
                    except ValueError:
                        print 'Could not interpret line %d\n%s' % (line,
                                                                   dbline)
                    rec_line += 1
            # Append last profile
            self.append(
                VelocityProfile(vp=num.array(vp), vs=num.array(vs),
                                h=num.array(h), d=num.array(depth),
                                lat=lat, lon=lon,
                                **meta)
            )
