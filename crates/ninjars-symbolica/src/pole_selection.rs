//! Map the public NinjaRS master basis onto oneloopmaster's pole residues.
use ninjars::{FloatLike, Laurent, MasterIntegral};
use oneloop::{PoleCoefficient, PoleSelection, ScalarIntegral};

pub(crate) fn select<T: FloatLike>(
    selection: PoleSelection,
    integral: MasterIntegral,
    values: &mut Laurent<T>,
) -> ninjars::Result<()> {
    if selection == PoleSelection::BOTH {
        return Ok(());
    }
    let coefficient = match integral {
        MasterIntegral::A0 => PoleCoefficient::Scalar(ScalarIntegral::A0),
        MasterIntegral::B0 => PoleCoefficient::Scalar(ScalarIntegral::B0),
        MasterIntegral::C0 => PoleCoefficient::Scalar(ScalarIntegral::C0),
        MasterIntegral::D0 => PoleCoefficient::Scalar(ScalarIntegral::D0),
        MasterIntegral::DB0 => PoleCoefficient::Scalar(ScalarIntegral::DB0),
        MasterIntegral::An { .. } => PoleCoefficient::TadpoleTensor,
        MasterIntegral::B1 => PoleCoefficient::BubbleMoment { rank: 1 },
        MasterIntegral::B11 => PoleCoefficient::BubbleMoment { rank: 2 },
        MasterIntegral::B111 => PoleCoefficient::BubbleMoment { rank: 3 },
        MasterIntegral::B1111 => PoleCoefficient::BubbleMoment { rank: 4 },
    };
    selection
        .apply(coefficient, values)
        .map_err(ninjars::Error::Symbolic)
}
