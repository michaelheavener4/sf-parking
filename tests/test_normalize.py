from sf_parking.normalize import normalize_meter, normalize_policy


def test_normalize_current_meter_row_uses_post_id_when_space_id_missing() -> None:
    meter = normalize_meter(
        {
            "active_meter_flag": "M",
            "post_id": "102-02990",
            "latitude": "37.7989",
            "longitude": "-122.4083",
            "street_name": "COLUMBUS AVE",
            "street_num": "415",
            "blockface_id": "363041",
        }
    )
    assert meter.parking_space_id is None
    assert meter.post_id == "102-02990"
    assert meter.active is True


def test_normalize_policy_handles_free_schedule_without_rate() -> None:
    policy = normalize_policy(
        {
            "dayofweek": "Mo",
            "startdate": "2026-07-13T00:00:00.000",
            "enddate": "2200-12-31T00:00:00.000",
            "starttime": "0:00",
            "endtime": "4:30",
            "parkingspaceid": "123238",
            "postid": "102-02990",
            "scheduletype": "FREE",
        }
    )
    assert policy.hourly_rate == 0
    assert policy.schedule_type == "FREE"
    assert policy.parking_space_id == 123238
