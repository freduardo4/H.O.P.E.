"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UpdateCalibrationDto = void 0;
const mapped_types_1 = require("@nestjs/mapped-types");
const create_calibration_dto_1 = require("./create-calibration.dto");
class UpdateCalibrationDto extends (0, mapped_types_1.PartialType)(create_calibration_dto_1.CreateCalibrationDto) {
}
exports.UpdateCalibrationDto = UpdateCalibrationDto;
//# sourceMappingURL=update-calibration.dto.js.map