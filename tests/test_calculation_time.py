"""How long a calculation took: printed, and written under its status line,
when its thread is done (PhysicsApp._start_calculation / _calculation_ended)."""
import pytest

import syncmoss.syncmoss_main as sm


def test_duration_text():
    assert sm._duration_text(12.34) == "12.3 s"
    assert sm._duration_text(245.4) == "4 min 05 s"
    assert sm._duration_text(2 * 3600 + 7 * 60 + 20) == "2 h 07 min"


class _Thread:
    def start(self):
        pass


@pytest.mark.gui
def test_the_time_follows_the_status_line(physics_app, capsys):
    physics_app.inprogress, physics_app.busy_with = True, 'Fitting'
    physics_app._start_calculation(_Thread())
    physics_app.set_status("Fit completed!", "green")
    physics_app._calculation_ended()
    status, took = physics_app.log.toPlainText().splitlines()
    assert status == "Fit completed!" and took.startswith("Fitting took ")
    assert took in capsys.readouterr().out
    assert not physics_app.inprogress
    physics_app._calculation_ended()      # once only: the clock is cleared
    assert physics_app.log.toPlainText().splitlines() == [status, took]
