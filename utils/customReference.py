"""
Created on Jun 8, 2023

@author: mning
"""
import numpy as np
from mne.defaults import DEFAULTS
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io.base import BaseRaw
from mne.io.constants import FIFF
from mne.io.meas_info import _check_ch_keys
from mne.io.pick import _ELECTRODE_CH_TYPES
from mne.io.pick import pick_channels
from mne.io.pick import pick_types
from mne.io.proj import setup_proj
from mne.utils import _check_option
from mne.utils import _check_preload
from mne.utils import _on_missing
from mne.utils import _validate_type
from mne.utils import logger
from mne.utils import verbose

"""
Derived from mne.io.reference.set_bipolar_reference.py, which is licensed
under the BSD 3-Clause "New" or "Revised" License.
"""


def _check_before_reference(inst, ref_from, ref_to, ch_type):
    """Prepare instance for referencing."""
    # Check to see that data is preloaded
    _check_preload(inst, "Applying a reference")

    ch_type = _get_ch_type(inst, ch_type)
    ch_dict = {**{type_: True for type_ in ch_type}, "meg": False, "ref_meg": False}
    eeg_idx = pick_types(inst.info, **ch_dict)

    if ref_to is None:
        ref_to = [inst.ch_names[i] for i in eeg_idx]
        extra = "EEG channels found"
    else:
        extra = "channels supplied"
    if len(ref_to) == 0:
        raise ValueError("No %s to apply the reference to" % (extra,))

    # After referencing, existing SSPs might not be valid anymore.
    projs_to_remove = []
    for i, proj in enumerate(inst.info["projs"]):
        # Remove any average reference projections
        if (
            proj["desc"] == "Average EEG reference"
            or proj["kind"] == FIFF.FIFFV_PROJ_ITEM_EEG_AVREF
        ):
            logger.info("Removing existing average EEG reference " "projection.")
            # Don't remove the projection right away, but do this at the end of
            # this loop.
            projs_to_remove.append(i)

        # Inactive SSPs may block re-referencing
        elif (
            not proj["active"]
            and len(
                [ch for ch in (ref_from + ref_to) if ch in proj["data"]["col_names"]]
            )
            > 0
        ):
            raise RuntimeError(
                "Inactive signal space projection (SSP) operators are "
                "present that operate on sensors involved in the desired "
                "referencing scheme. These projectors need to be applied "
                "using the apply_proj() method function before the desired "
                "reference can be set."
            )

    for i in projs_to_remove:
        del inst.info["projs"][i]

    # Need to call setup_proj after changing the projs:
    inst._projector, _ = setup_proj(inst.info, add_eeg_ref=False, activate=False)

    # If the reference touches EEG/ECoG/sEEG/DBS electrodes, note in the
    # info that a non-CAR has been applied.
    ref_to_channels = pick_channels(inst.ch_names, ref_to, ordered=True)
    if len(np.intersect1d(ref_to_channels, eeg_idx)) > 0:
        with inst.info._unlock():
            inst.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

    return ref_to


def _get_ch_type(inst, ch_type):
    _validate_type(ch_type, (str, list, tuple), "ch_type")
    valid_ch_types = ("auto",) + _ELECTRODE_CH_TYPES
    if isinstance(ch_type, str):
        _check_option("ch_type", ch_type, valid_ch_types)
        if ch_type != "auto":
            ch_type = [ch_type]
    elif isinstance(ch_type, (list, tuple)):
        for type_ in ch_type:
            _validate_type(type_, str, "ch_type")
            _check_option("ch_type", type_, valid_ch_types[1:])
        ch_type = list(ch_type)

    # if ch_type is 'auto', search through list to find first reasonable
    # reference-able channel type.
    if ch_type == "auto":
        for type_ in _ELECTRODE_CH_TYPES:
            if type_ in inst:
                ch_type = [type_]
                logger.info(
                    "%s channel type selected for "
                    "re-referencing" % DEFAULTS["titles"][type_]
                )
                break
        # if auto comes up empty, or the user specifies a bad ch_type.
        else:
            raise ValueError(
                "No EEG, ECoG, sEEG or DBS channels found " "to rereference."
            )
    return ch_type


_ref_dict = {
    FIFF.FIFFV_MNE_CUSTOM_REF_ON: "on",
    FIFF.FIFFV_MNE_CUSTOM_REF_OFF: "off",
    FIFF.FIFFV_MNE_CUSTOM_REF_CSD: "CSD",
}


def _check_can_reref(inst):
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), "Instance")
    current_custom = inst.info["custom_ref_applied"]
    if current_custom not in (
        FIFF.FIFFV_MNE_CUSTOM_REF_ON,
        FIFF.FIFFV_MNE_CUSTOM_REF_OFF,
    ):
        raise RuntimeError(
            "Cannot set new reference on data with custom "
            "reference type %r" % (_ref_dict[current_custom],)
        )


@verbose
def set_egc_reference(
    inst,
    anode,
    cathode,
    ch_name=None,
    ch_info=None,
    drop_refs=True,
    copy=True,
    on_bad="warn",
    verbose=None,
):
    """Re-reference selected channels using a bipolar referencing scheme.

    A bipolar reference takes the difference between two channels (the anode
    minus the cathode) and adds it as a new virtual channel. The original
    channels will be dropped by default.

    Multiple anodes and cathodes can be specified, in which case multiple
    virtual channels will be created. The 1st cathode will be subtracted
    from the 1st anode, the 2nd cathode from the 2nd anode, etc.

    By default, the virtual channels will be annotated with channel-info and
    -location of the anodes and coil types will be set to EEG_BIPOLAR.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Data containing the unreferenced channels.
    anode : str | list of str
        The name(s) of the channel(s) to use as anode in the bipolar reference.
    cathode : str | list of str
        The name(s) of the channel(s) to use as cathode in the bipolar
        reference.
    ch_name : str | list of str | None
        The channel name(s) for the virtual channel(s) containing the resulting
        signal. By default, bipolar channels are named after the anode and
        cathode, but it is recommended to supply a more meaningful name.
    ch_info : dict | list of dict | None
        This parameter can be used to supply a dictionary (or a dictionary for
        each bipolar channel) containing channel information to merge in,
        overwriting the default values. Defaults to None.
    drop_refs : bool
        Whether to drop the anode/cathode channels from the instance.
    copy : bool
        Whether to operate on a copy of the data (True) or modify it in-place
        (False). Defaults to True.
    on_bad : str
        If a bipolar channel is created from a bad anode or a bad cathode, mne
        warns if on_bad="warns", raises ValueError if on_bad="raise", and does
        nothing if on_bad="ignore". For "warn" and "ignore", the new bipolar
        channel will be marked as bad. Defaults to on_bad="warns".
    %(verbose)s

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with the specified channels re-referenced.

    See Also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.

    Notes
    -----
    1. If the anodes contain any EEG channels, this function removes
       any pre-existing average reference projections.

    2. During source localization, the EEG signal should have an average
       reference.

    3. The data must be preloaded.

    .. versionadded:: 0.9.0
    """
    from mne.io.meas_info import create_info
    from mne.io import RawArray
    from mne.epochs import EpochsArray
    from mne.evoked import EvokedArray

    _check_can_reref(inst)
    if not isinstance(anode, list):
        anode = [anode]

    if not isinstance(cathode, list):
        cathode = [cathode]

    if len(anode) != len(cathode):
        raise ValueError(
            "Number of anodes (got %d) must equal the number "
            "of cathodes (got %d)." % (len(anode), len(cathode))
        )

    if ch_name is None:
        ch_name = [f"{a}+{c}" for (a, c) in zip(anode, cathode)]
    elif not isinstance(ch_name, list):
        ch_name = [ch_name]
    if len(ch_name) != len(anode):
        raise ValueError(
            "Number of channel names must equal the number of "
            "anodes/cathodes (got %d)." % len(ch_name)
        )

    # Check for duplicate channel names (it is allowed to give the name of the
    # anode or cathode channel, as they will be replaced).
    for ch, a, c in zip(ch_name, anode, cathode):
        if ch not in [a, c] and ch in inst.ch_names:
            raise ValueError(
                'There is already a channel named "%s", please '
                "specify a different name for the bipolar "
                "channel using the ch_name parameter." % ch
            )

    if ch_info is None:
        ch_info = [{} for _ in anode]
    elif not isinstance(ch_info, list):
        ch_info = [ch_info]
    if len(ch_info) != len(anode):
        raise ValueError(
            "Number of channel info dictionaries must equal the "
            "number of anodes/cathodes."
        )

    if copy:
        inst = inst.copy()

    anode = _check_before_reference(
        inst, ref_from=cathode, ref_to=anode, ch_type="auto"
    )

    # Create bipolar reference channels by multiplying the data
    # (channels x time) with a matrix (n_virtual_channels x channels)
    # and add them to the instance.
    multiplier = np.zeros((len(anode), len(inst.ch_names)))
    for idx, (a, c) in enumerate(zip(anode, cathode)):
        multiplier[idx, inst.ch_names.index(a)] = 1
        multiplier[idx, inst.ch_names.index(c)] = 1

    ref_info = create_info(
        ch_names=ch_name,
        sfreq=inst.info["sfreq"],
        ch_types=inst.get_channel_types(picks=anode),
    )

    # Update "chs" in Reference-Info.
    for ch_idx, (an, info) in enumerate(zip(anode, ch_info)):
        _check_ch_keys(info, ch_idx, name="ch_info", check_min=False)
        an_idx = inst.ch_names.index(an)
        # Copy everything from anode (except ch_name).
        an_chs = {k: v for k, v in inst.info["chs"][an_idx].items() if k != "ch_name"}
        ref_info["chs"][ch_idx].update(an_chs)
        # Set coil-type to bipolar.
        ref_info["chs"][ch_idx]["coil_type"] = FIFF.FIFFV_COIL_EEG_BIPOLAR
        # Update with info from ch_info-parameter.
        ref_info["chs"][ch_idx].update(info)

    # Set other info-keys from original instance.
    pick_info = {
        k: v
        for k, v in inst.info.items()
        if k not in ["chs", "ch_names", "bads", "nchan", "sfreq"]
    }

    with ref_info._unlock():
        ref_info.update(pick_info)

    # Rereferencing of data.
    ref_data = multiplier @ inst._data

    if isinstance(inst, BaseRaw):
        ref_inst = RawArray(ref_data, ref_info, first_samp=inst.first_samp, copy=None)
    elif isinstance(inst, BaseEpochs):
        ref_inst = EpochsArray(
            ref_data,
            ref_info,
            events=inst.events,
            tmin=inst.tmin,
            event_id=inst.event_id,
            metadata=inst.metadata,
        )
    else:
        ref_inst = EvokedArray(
            ref_data,
            ref_info,
            tmin=inst.tmin,
            comment=inst.comment,
            nave=inst.nave,
            kind="average",
        )

    # Add referenced instance to original instance.
    inst.add_channels([ref_inst], force_update_info=True)

    # Handle bad channels.
    bad_bipolar_chs = []
    for ch_idx, (a, c) in enumerate(zip(anode, cathode)):
        if a in inst.info["bads"] or c in inst.info["bads"]:
            bad_bipolar_chs.append(ch_name[ch_idx])

    # Add warnings if bad channels are present.
    if bad_bipolar_chs:
        msg = f"Bipolar channels are based on bad channels: {bad_bipolar_chs}."
        _on_missing(on_bad, msg)
        inst.info["bads"] += bad_bipolar_chs

    added_channels = ", ".join([name for name in ch_name])
    logger.info(f"Added the following bipolar channels:\n{added_channels}")

    for attr_name in ["picks", "_projector"]:
        setattr(inst, attr_name, None)

    # Drop remaining channels.
    if drop_refs:
        drop_channels = list((set(anode) | set(cathode)) & set(inst.ch_names))
        inst.drop_channels(drop_channels)

    return inst
