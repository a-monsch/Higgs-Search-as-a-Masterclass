"""
    Helper functions for the masterclass
"""

import warnings
from itertools import combinations
import shutil
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import vector as vec

if shutil.which('latex'):
    matplotlib.rc("font", size=16, family="serif")
    matplotlib.rcParams["text.usetex"] = True

lw = 2
prev_hists: Dict[str, list] = {"z1": [], "z2": [], "four_lep": [], "filter": []}


def clear_hist_plots(df: pd.DataFrame) -> pd.DataFrame:
    global prev_hists
    prev_hists = {"z1": [], "z2": [], "four_lep": [], "filter": []}
    return df


def plot_masses(
    df: pd.DataFrame,
    mass_function: Callable,
    filter_name: str = "",
    z1_bins: int = 100,
    z2_bins: int = 100,
    four_lep_bins: int = 100,
    z1_range: Tuple[float, float] = (0, 139),
    z2_range: Tuple[float, float] = (0, 79),
    four_lep_range: Tuple[float, float] = (100, 149),
    yscale: str = "symlog",
    show_z1_and_z2_mass: bool = True,
    yedge: Union[None, float, int] = None,
    xdim: Union[int, float] = 14,
    ydim: Union[int, float] = 3,
    title: Union[None, str] = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    hist_z1, edges_z1 = np.histogram(
        mass_function(df["Z1"]["E"], df["Z1"]["px"], df["Z1"]["py"], df["Z1"]["pz"]),
        bins=z1_bins,
        range=z1_range,
    )

    hist_z2, edges_z2 = np.histogram(
        mass_function(df["Z2"]["E"], df["Z2"]["px"], df["Z2"]["py"], df["Z2"]["pz"]),
        bins=z2_bins,
        range=z2_range,
    )

    hist_four_lep, edges_four_lep = np.histogram(
        mass_function(df["four_lep"]["E"], df["four_lep"]["px"], df["four_lep"]["py"], df["four_lep"]["pz"]),
        bins=four_lep_bins,
        range=four_lep_range,
    )

    ymax = []
    global prev_hists

    fig, ax = plt.subplots(2 if yedge is not None else 1, 3 if show_z1_and_z2_mass else 1, figsize=(xdim, ydim))
    plt.subplots_adjust(wspace=0, hspace=0)

    if show_z1_and_z2_mass and (yedge is not None):
        pass
    elif show_z1_and_z2_mass and (yedge is None):
        ax = [ax]
    elif (not show_z1_and_z2_mass) and (yedge is not None):
        ax = [[it] for it in ax]
    else:
        ax = [[ax]]

    if len(prev_hists["z1"]) > 0:
        for num, _ in list(enumerate(prev_hists["z1"])):
            for idx in [0, 1] if yedge else [0]:
                ax[idx][0].step(
                    prev_hists["four_lep"][num][1],
                    np.pad(prev_hists["four_lep"][num][0], (0, 1)),
                    where="post",
                    label=prev_hists["filter"][num] if idx == 0 else None,
                    alpha=0.5,
                    lw=lw,
                )
                if show_z1_and_z2_mass:
                    ax[idx][1].step(
                        prev_hists["z1"][num][1],
                        np.pad(prev_hists["z1"][num][0], (0, 1)),
                        where="post",
                        alpha=0.5,
                        lw=lw,
                    )

                    ax[idx][2].step(
                        prev_hists["z2"][num][1],
                        np.pad(prev_hists["z2"][num][0], (0, 1)),
                        where="post",
                        alpha=0.5,
                        lw=lw,
                    )

            ymax.extend(
                [
                    prev_hists["z1"][num][0].max(),
                    prev_hists["z2"][num][0].max(),
                    prev_hists["four_lep"][num][0].max(),
                ],
            )

    for idx in [0, 1] if yedge else [0]:
        ax[idx][0].step(
            edges_four_lep,
            np.pad(hist_four_lep, (0, 1)),
            where="post",
            color="black",
            label=f"Massenverteilung: {filter_name}" if idx == 0 else None,
            lw=lw,
        )
        if show_z1_and_z2_mass:
            ax[idx][1].step(
                edges_z1,
                np.pad(hist_z1, (0, 1)),
                where="post",
                color="black",
                lw=lw,
            )

            ax[idx][2].step(
                edges_z2,
                np.pad(hist_z2, (0, 1)),
                where="post",
                color="black",
                lw=lw,
            )

    prev_hists["z1"].append((hist_z1, edges_z1))
    prev_hists["z2"].append((hist_z2, edges_z2))
    prev_hists["four_lep"].append((hist_four_lep, edges_four_lep))
    prev_hists["filter"].append(filter_name or f"Filter {len(prev_hists['filter'])}")
    ymax.extend([hist_z1.max(), hist_z2.max(), hist_four_lep.max()])

    if yedge is None:
        ax[0][0].set(
            title="Kombinierte vier-Leptonen",
            ylabel="N",
            ylim=(0, max(ymax) * 1.2),
            xlim=(edges_four_lep.min(), edges_four_lep.max()),
            xlabel=r"$m_{4\ell}$ in GeV",
            yscale=yscale,
        )
        if show_z1_and_z2_mass:
            ax[0][1].set(
                title="$Z_1$-Boson Kandidaten",
                ylim=(0, max(ymax) * 1.2),
                xlim=(edges_z1.min(), edges_z1.max()),
                xlabel=r"$m_{Z_1}$ in GeV",
                yscale=yscale,
            )

            ax[0][2].set(
                title="$Z_2$-Boson Kandidaten",
                ylim=(0, max(ymax) * 1.2),
                xlim=(edges_z2.min(), edges_z2.max()),
                xlabel=r"$m_{Z_2}$ in GeV",
                yscale=yscale,
            )

        ax[0][0].grid(axis="y")
        if show_z1_and_z2_mass:
            ax[0][1].grid(axis="y")
            ax[0][1].yaxis.set_ticks(ax[0][1].get_yticks())
            ax[0][1].set_yticklabels(["" for _ in ax[0][1].get_yticks()])
            ax[0][2].grid(axis="y")
            ax[0][2].yaxis.set_ticks(ax[0][2].get_yticks())
            ax[0][2].set_yticklabels(["" for _ in ax[0][1].get_yticks()])

        ax = ax[0]
        fig.legend(
            bbox_to_anchor=(
                ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).x0 / fig.bbox_inches.width,
                0.115 + ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).y1 / fig.bbox_inches.height,
                sum(_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width / fig.bbox_inches.width for _ax in ax),
                0.102,
            ),
            loc=3,
            ncol=1,
            mode="expand",
            borderaxespad=0.0,
            prop={"size": 16},
        )
    else:
        ax[0][0].text(
            -0.11,
            0.01,
            "N",
            horizontalalignment="right",
            verticalalignment="bottom",
            rotation=90,
            transform=ax[0][0].transAxes,
        )

        ax[0][0].set(
            title="Kombinierte vier-Leptonen",
            ylim=(yedge, max(ymax) * 1.2),
            xlim=(edges_four_lep.min(), edges_four_lep.max()),
            xticks=[],
            yscale="log",
        )
        if show_z1_and_z2_mass:
            ax[0][1].set(
                title="$Z_1$-Boson Kandidaten",
                ylim=(yedge, max(ymax) * 1.2),
                xlim=(edges_z1.min(), edges_z1.max()),
                xlabel=r"$m_{Z_2}$ in GeV",
                xticks=[],
                yscale="log",
            )

            ax[0][2].set(
                title="$Z_2$-Boson Kandidaten",
                ylim=(yedge, max(ymax) * 1.2),
                xlim=(edges_z2.min(), edges_z2.max()),
                xticks=[],
                yscale="log",
            )

        # ---

        ax[1][0].set(
            ylim=(0, yedge),
            xlim=(edges_four_lep.min(), edges_four_lep.max()),
            xlabel=r"$m_{4\ell}$ in GeV",
        )
        if show_z1_and_z2_mass:
            ax[1][1].set(
                ylim=(0, yedge),
                xlim=(edges_z1.min(), edges_z1.max()),
                xlabel=r"$m_{Z_1}$ in GeV",
            )

            ax[1][2].set(
                ylim=(0, yedge),
                xlim=(edges_z2.min(), edges_z2.max()),
                xlabel=r"$m_{Z_2}$ in GeV",
            )

        for i in range(2):
            for j in range(3 if show_z1_and_z2_mass else 1):
                ax[i][j].grid(axis="y")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if show_z1_and_z2_mass:
                ax[0][1].set_yticklabels(["" for _ in ax[0][1].get_yticks()])
                ax[1][1].set_yticklabels(["" for _ in ax[1][1].get_yticks()])
                ax[0][2].set_yticklabels(["" for _ in ax[0][2].get_yticks()])
                ax[1][2].set_yticklabels(["" for _ in ax[1][2].get_yticks()])

        fig.legend(
            bbox_to_anchor=(
                ax[0][0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).x0 / fig.bbox_inches.width,
                0.115 + ax[0][0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).y1 / fig.bbox_inches.height,
                sum(_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width / fig.bbox_inches.width for _ax in ax[0]),
                0.102,
            ),
            loc=3,
            ncol=1,
            mode="expand",
            borderaxespad=0.0,
            prop={"size": 14},
        )

    if title is not None:
        fig.suptitle(title, y=1.0 + 0.2 * len(prev_hists["z1"]))

    if show_plot:
        plt.show()
    else:
        plt.close("all")

    return df


# Class to access particle four vectors
@pd.api.extensions.register_dataframe_accessor("v4")
class FourVecAccessor(object):
    def __init__(self, pandas_obj: Union[pd.Series, pd.DataFrame]) -> None:
        # to distinguish between multiple particles or single particle
        # we only need to save the column information,
        self._obj_columns = pandas_obj.columns
        # to keep data consistent when appending columns unsing this accessor save indices to use them in returns
        self._obj_indices = pandas_obj.index
        # get the correct index level, 0 for single particle, 1 for multiple
        _vars = self._obj_columns.get_level_values(self._obj_columns.nlevels - 1)

        if "E" in _vars and "px" in _vars:
            kin_vars = ["E", "px", "py", "pz"]
        elif "E" in _vars and "pt" in _vars:
            kin_vars = ["E", "pt", "phi", "eta"]
        else:
            raise KeyError("No matching structure implemented for interpreting the data as a four " "momentum!")

        # the following lines are where the magic happens

        # no multi-index, just on particle
        if self._obj_columns.nlevels == 1:
            # get the dtypes for the kinetic variables
            dtypes = pandas_obj.dtypes
            kin_view = list(map(lambda x: (x, dtypes[x]), kin_vars))

            # get the kinetic variables from the dataframe and convert it to a numpy array.
            # require it to be C_CONTIGUOUS, vector uses C-Style
            # This array can then be viewed as a vector object.
            # Every property not given is calculated on the fly by the vector object.
            # E.g. the mass is not stored but calculated when the energy is given and vice versa.
            self._v4 = np.require(pandas_obj[kin_vars].to_numpy(), requirements="C").view(kin_view).view(vec.MomentumNumpy4D)

        # multi-index, e.g. getting the four momentum for multiple particles
        elif self._obj_columns.nlevels == 2:
            # get the dtypes for the kinetic variables
            # assume the same dtypes for the other particles
            dtypes = pandas_obj[self._obj_columns.get_level_values(0).unique()[0]].dtypes
            kin_view = list(map(lambda x: (x, dtypes[x]), kin_vars))
            self._v4 = (
                np.require(pandas_obj.loc[:, (self._obj_columns.get_level_values(0).unique(), kin_vars)].to_numpy(), requirements="C")
                .view(kin_view)
                .view(vec.MomentumNumpy4D)
            )

        else:
            raise IndexError("Expected a dataframe with a maximum of two multi-index levels.")

    def __getattribute__(self, item: str) -> Any:
        """
        Attributes of this accessor are forwarded to the four vector.

        Returns either a pandas dataframe, if we have multiple particles
        or a pandas Series for a single particle.
        """
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            try:
                return pd.DataFrame(
                    self._v4.__getattribute__(item),
                    columns=pd.MultiIndex.from_product([self._obj_columns.unique(0), [item]]),
                    index=self._obj_indices,
                )
            except ValueError:
                try:
                    return pd.Series(self._v4.__getattribute__(item).flatten(), name=item, index=self._obj_indices)
                except AttributeError as e:
                    if "'function' object has no attribute 'flatten'" in str(e):
                        raise AttributeError(
                            "Functions of the four vectors can NOT be called directly via the "
                            "accessor. Use the vector property instead! "
                            "Usage: 'df['particle'].v4.vector.desired_function()'"
                        )
                    raise e

    @property
    def vector(self) -> Union[pd.Series, pd.DataFrame]:
        """The four vector object itself. It's required when using methods like boosting."""
        if self._obj_columns.nlevels == 1:
            return self._v4[:, 0]
        else:
            return self._v4


# Helper function to reindex a changed dataframe to avoid unexpected behaviour
# Can be used via df.pipe(reset_idx)
def reset_idx(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index().drop(columns=["index"], axis=0, level=0)


# Basic Event and Lepton Filter, simplified for .pipe usage


class _FilterBase(object):
    # Helper 1
    @staticmethod
    def _empty_mask(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a True-filled mask of top level objects of a given pd.DataFrame

        :param df: pd.DataFrame
        :return: pd.DataFrame (mask)
        """
        return pd.DataFrame(np.ones((df.shape[0], df.columns.levshape[0])), columns=df.columns.levels[0], dtype=bool)

    # Helper 2
    @staticmethod
    def _particle_filter_mask(df: pd.DataFrame, mask: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Creates a full mask of a pd.DataFrame out of a given mask of top level object DataFrame
        refilling np.nans with True if some are present.

        :param df: pd.DataFrame
        :param mask: pd.DataFrame of top levels
        :return: pd.DataFrame (mask)
        """
        return (mask).reindex(df.columns, level=0, axis=1).fillna(True)

    # Helper 3
    @staticmethod
    def _is_flavour(df: pd.DataFrame, obj: List[str], flavour: int) -> pd.DataFrame:
        """
        Creates a boolean top level mask considering the lepton flavour

        :param df: pd.DataFrame
        :param obj: List of strings of top level objects, i.e. ["lepton_0", "lepton_1", ...]
        :param flavour: 0 or 1; (muon or electron)
        :return: pd.DataFrame (mask)
        """
        return (df.loc[:, (obj, "flavour")] == flavour).droplevel(1, axis=1)

    # Helper 4
    @staticmethod
    def _leptons(df: pd.DataFrame) -> List:
        return np.unique([it for it in df.columns.get_level_values(0) if "lepton" in it]).tolist()


class _EventFilterMasks(_FilterBase):
    @staticmethod
    def _min_lepton_number(df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Performs a check on the minimum number of leptons (with corresponding
        flavour) within an event and creates a pd.Series 1D mask

        :param df: pd.DataFrame
        :return: pd.Series (mask)
        """

        # small helper function that summarize three steps:
        # 1. Mask all leptons with the undesired falvour with False/0
        # 2. Counts the remaining number of leptons with the desired flavour (.sum(axis=1)) and compares
        #    it with a given minimum number of leptons (n)
        # 3. Checks if the event channel corresponds to a given channel

        leptons = _FilterBase._leptons(df)

        def _get_submask(flavour: int, n: int, channel: int) -> Union[pd.Series, pd.DataFrame]:
            _min_leps_mask = _FilterBase._is_flavour(df, leptons, flavour).sum(axis=1) >= n
            return _min_leps_mask & (df.event_information.channel == channel)

        tmp_mask = pd.Series(np.zeros_like(df.event_information.channel), index=df.index).astype(bool)

        tmp_mask |= _get_submask(0, 4, 0)  # four muon channel
        tmp_mask |= _get_submask(1, 4, 1)  # four electron channel
        tmp_mask |= _get_submask(0, 2, 2) & _get_submask(1, 2, 2)  # mixed channel

        return tmp_mask

    @staticmethod
    def _neutral_charge(df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Checking all the possible combinations of two (four) leptons to see if there
        is a charge neutral combination in the events that could be used to
        reconstruct the Z boson(s).

        :param df: pd.DataFrame
        :return: pd.Series (mask)
        """

        def is_channel(channel: int) -> pd.Series:
            return df.event_information.channel == channel

        def get_charge(combination: list) -> Union[pd.Series, pd.DataFrame]:
            return df.loc[:, (combination, "charge")].droplevel(1, axis=1)

        tmp_mask = pd.Series(np.zeros_like(df.event_information.channel), index=df.index).astype(bool)

        leptons = _FilterBase._leptons(df)

        for lepton_combination in combinations(leptons, 4):
            leps = list(lepton_combination)  # converting from tuple
            charge = get_charge(leps)  # charge for this combination
            muons, electrons = _FilterBase._is_flavour(df[leps], leps, 0), _FilterBase._is_flavour(df[leps], leps, 1)  # flavour masks

            # eventwise summation of charges (axis=1) given a specific combination
            four_mu = (charge[muons].sum(axis=1) == 0) & is_channel(0)
            four_el = (charge[electrons].sum(axis=1) == 0) & is_channel(1)
            two_mu_two_el = (charge[muons].sum(axis=1) == 0) & (charge[electrons].sum(axis=1) == 0) & is_channel(2)

            tmp_mask |= four_mu | four_el | two_mu_two_el

        return tmp_mask

    @staticmethod
    def _z_masses(
        df: pd.DataFrame,
        z1_mass_min: Union[float, int],
        z1_mass_max: Union[float, int],
        z2_mass_min: Union[float, int],
        z2_mass_max: Union[float, int],
    ) -> pd.Series:
        """
        Performs a check on the masses of calculated Z Bosons
        within an event and creates a pd.Series 1D mask

        :param df: pd.DataFrame
        :return: pd.Series (mask)
        """

        tmp_mask = pd.Series(np.ones_like(df.event_information.channel), index=df.index).astype(bool)
        tmp_mask &= (df.Z1.v4.mass > z1_mass_min) & (df.Z1.v4.mass < z1_mass_max)
        tmp_mask &= (df.Z2.v4.mass > z2_mass_min) & (df.Z2.v4.mass < z2_mass_max)

        return tmp_mask


class _LeptonFilterMasks(_FilterBase):
    @staticmethod
    def _min_pt(
        df: pd.DataFrame,
        min_pt_muon: Union[int, float] = 5,
        min_pt_electron: Union[int, float] = 7,
    ) -> Union[pd.DataFrame, pd.Series]:
        leptons = _FilterBase._leptons(df)
        pt = df[leptons].v4.pt.droplevel(1, axis=1)
        muons, electrons = _FilterBase._is_flavour(df, leptons, 0), _FilterBase._is_flavour(df, leptons, 1)

        tmp_mask = _FilterBase._empty_mask(df)
        tmp_mask &= ((pt > min_pt_electron) & electrons) | ((pt > min_pt_muon) & muons)

        return _FilterBase._particle_filter_mask(df, tmp_mask)

    @staticmethod
    def _relative_isolation(
        df: pd.DataFrame,
        relative_isolation_value: Union[int, float] = 0.4,
    ) -> Union[pd.DataFrame, pd.Series]:
        leptons = _FilterBase._leptons(df)
        relative_isolation = df.loc[:, (leptons, "relpfiso")].droplevel(1, axis=1)

        tmp_mask = _FilterBase._empty_mask(df)
        tmp_mask &= relative_isolation < relative_isolation_value

        return _FilterBase._particle_filter_mask(df, tmp_mask)


# Provided filter colelction that is used for masterclass
class Filter(_LeptonFilterMasks, _EventFilterMasks):
    # for .pipe Method
    def relative_isolation_of_lepton(df: pd.DataFrame, relative_isolation_value: Union[int, float]) -> pd.DataFrame:
        return df[_LeptonFilterMasks._relative_isolation(df, relative_isolation_value)].pipe(reset_idx)

    # For .pipe Method
    @staticmethod
    def min_pt_of_lepton(df: pd.DataFrame, min_pt_muon: Union[int, float], min_pt_electron: Union[int, float]) -> pd.DataFrame:
        return df[_LeptonFilterMasks._min_pt(df, min_pt_muon, min_pt_electron)].pipe(reset_idx)

    # for .pipe Method
    @staticmethod
    def z_masses(
        df: pd.DataFrame,
        z1_mass_min: Union[int, float],
        z1_mass_max: Union[int, float],
        z2_mass_min: Union[int, float],
        z2_mass_max: Union[int, float],
    ) -> pd.DataFrame:
        return df[_EventFilterMasks._z_masses(df, z1_mass_min, z1_mass_max, z2_mass_min, z2_mass_max).to_numpy()].pipe(reset_idx)

    # for .pipe Method
    @staticmethod
    def min_lepton_number(df: pd.DataFrame) -> pd.DataFrame:
        return df[_EventFilterMasks._min_lepton_number(df).to_numpy()].pipe(reset_idx)

    # for .pipe Method
    @staticmethod
    def neutral_charge(df: pd.DataFrame) -> pd.DataFrame:
        return df[_EventFilterMasks._neutral_charge(df).to_numpy()].pipe(reset_idx)


# Helper function for getting the lepton names of muons and electrons in an event with a two muons and two electrons decay
def get_lepton_names_by_flavour(row: pd.Series, flavour: int = 0) -> Tuple:
    leptons = _FilterBase._leptons(pd.DataFrame(row).T)
    _filter = np.array(leptons)[_FilterBase._is_flavour(pd.DataFrame(row).T, leptons, flavour).values[0]].tolist()
    return tuple(_filter)


# Helper Function for plotting the comparison between Monte Carlo simulation and the
# actual measurement. A channel wise MC scaling is performed during the creation
def _get_scaled_bins_mc_data_comparison(
    df_data: pd.DataFrame,
    df_mc_sig: pd.DataFrame,
    df_mc_bkg: pd.DataFrame,
    obj: str = "four_lep",
    quantity: str = "mass",
    bins: int = 15,
    hist_range: Tuple[float, float] = (106, 151),
    scaling_df_file: str = "data/histogram_mc_scaling.csv",
) -> Tuple:
    """
    Helper Function to create the scaled histogram bins from given dataframes containing MC simulation

    :param df_data: pd.DataFrame containing the measurement
    :param df_mc_sig: pd.DataFrame containing the simulated signal
    :param df_mc_bkg: pd.DataFrame containing the simulated background
    :param obj: str name of a given top-level physics object that is present in all
                dataframes, e.g. "Z1", "four_lep", "lepton_0"
    :return: tuple of created signal_bins, background_bins, measurement_bins,
             corresponding bin_edges and the middle coordinates of the bins (measurement_x)
    """

    # Load Information for histogram scaling
    scale_df = pd.read_csv(scaling_df_file)

    # create empty arrays for a histogram
    bins_sig, edges = np.histogram(np.array([]), bins=bins, range=hist_range, weights=np.array([]))
    bins_bkg = np.zeros_like(bins_sig)

    for _channel in [0, 1, 2]:  # creating the MC histograms channel wise
        # Helper for getting specific attribute and the histogram scale factor
        def get_attr(df: pd.DataFrame) -> npt.NDArray:
            return getattr(df[df.event_information.channel == _channel][obj].v4, quantity).to_numpy()

        def get_factor(process: str) -> npt.NDArray:
            return scale_df[(scale_df.process == process) & (scale_df.channel == _channel)].f.to_numpy()

        # summation for the final histogram
        bins_sig += np.histogram(get_attr(df_mc_sig), bins=edges)[0].astype(float) * get_factor("signal")
        bins_bkg += np.histogram(get_attr(df_mc_bkg), bins=edges)[0].astype(float) * get_factor("background")

    bins_measurement, _ = np.histogram(getattr(df_data[[obj]].v4, quantity), bins=edges)

    bins_sig = np.pad(bins_sig, 1)[1:]  # padding for .fill_between plotting method
    bins_bkg = np.pad(bins_bkg, 1)[1:]  # padding for .fill_between plotting method
    measurement_x = edges[1:] - abs(edges[1] - edges[0]) / 2  # for plt.errorbar method

    return bins_sig, bins_bkg, bins_measurement, edges, measurement_x


def plot_mc_data_comparison(
    df_data: pd.DataFrame,
    df_mc_sig: pd.DataFrame,
    df_mc_bkg: pd.DataFrame,
    bins: int = 15,
    hist_range: Tuple[float, float] = (106, 151),
    xlabel: str = r"$m_{4\ell}$ in GeV",
) -> Tuple:
    bins_sig, bins_bkg, bins_data, edges, x_data = _get_scaled_bins_mc_data_comparison(
        df_data,
        df_mc_sig,
        df_mc_bkg,
        bins=bins,
        hist_range=hist_range,
    )

    # Plotting starts here
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.errorbar(x_data, bins_data, yerr=np.sqrt(bins_data), fmt="ko", label="Messung")
    ax.fill_between(
        edges,
        bins_sig + bins_bkg,
        bins_bkg,
        step="post",
        color="none",
        label=r"Signal (MC) ($m_{\mathrm{H}}=125\, \mathrm{GeV}$)",
        lw=2,
        facecolor="none",
        edgecolor="orangered",
        hatch="//",
    )
    ax.fill_between(
        edges,
        bins_bkg,
        step="post",
        color="royalblue",
        label="Untergrund (MC)",
    )

    ax.set(
        xlabel=xlabel,
        ylabel=f"N/{round((edges[-1] - edges[0]) / len(edges[1:]), 1)} GeV",
        xlim=(edges[0], edges[-1]),
        ylim=(0, None),
    )

    ax.legend()

    plt.tight_layout()

    plt.show()

    return (
        bins_bkg[:-1],  # without padding for plotting (tailing zero)
        bins_data,
    )
