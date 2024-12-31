// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

contract AttendanceTracker {
    struct Attendance {
        string status; // "in" or "out"
        uint256 timestamp;
    }

    struct Movement {
        uint256 exitTime;
        uint256 returnTime;
    }

    // Storage mappings for attendance and movements
    mapping(bytes32 => Attendance[]) private attendanceRecords;
    mapping(bytes32 => Movement[]) private movementRecords;

    // Events for logging changes
    event AttendanceUpdated(string indexed studentName, string status, uint256 timestamp);
    event MovementLogged(string indexed studentName, uint256 exitTime, uint256 returnTime);

    /**
     * @notice Updates the attendance of a student (mark "in" or "out").
     * @param _studentName The name of the student.
     * @param _status The attendance status ("in" or "out").
     */
    function updateAttendance(string memory _studentName, string memory _status) external {
        require(
            keccak256(bytes(_status)) == keccak256(bytes("in")) || 
            keccak256(bytes(_status)) == keccak256(bytes("out")),
            "Invalid status"
        );

        bytes32 nameHash = keccak256(abi.encodePacked(_studentName));

        Attendance memory newAttendance = Attendance({
            status: _status,
            timestamp: block.timestamp
        });

        attendanceRecords[nameHash].push(newAttendance);
        emit AttendanceUpdated(_studentName, _status, block.timestamp);
    }

    /**
     * @notice Logs the movement of a student (exit and return times).
     * @param _studentName The name of the student.
     * @param _exitTime The timestamp when the student exited.
     * @param _returnTime The timestamp when the student returned.
     */
    function logMovement(string memory _studentName, uint256 _exitTime, uint256 _returnTime) external {
        require(_exitTime < _returnTime, "Exit time must be before return time");

        bytes32 nameHash = keccak256(abi.encodePacked(_studentName));

        Movement memory newMovement = Movement({
            exitTime: _exitTime,
            returnTime: _returnTime
        });

        movementRecords[nameHash].push(newMovement);
        emit MovementLogged(_studentName, _exitTime, _returnTime);
    }

    /**
     * @notice Retrieves the attendance records for a student.
     * @param _studentName The name of the student.
     * @return A list of attendance records.
     */
    function getAttendanceRecords(string memory _studentName) external view returns (Attendance[] memory) {
        bytes32 nameHash = keccak256(abi.encodePacked(_studentName));
        return attendanceRecords[nameHash];
    }

    /**
     * @notice Retrieves the movement records for a student.
     * @param _studentName The name of the student.
     * @return A list of movement records.
     */
    function getMovementRecords(string memory _studentName) external view returns (Movement[] memory) {
        bytes32 nameHash = keccak256(abi.encodePacked(_studentName));
        return movementRecords[nameHash];
    }
}
