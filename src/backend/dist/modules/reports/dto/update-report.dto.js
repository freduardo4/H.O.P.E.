"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UpdateReportDto = void 0;
const mapped_types_1 = require("@nestjs/mapped-types");
const create_report_dto_1 = require("./create-report.dto");
class UpdateReportDto extends (0, mapped_types_1.PartialType)(create_report_dto_1.CreateReportDto) {
}
exports.UpdateReportDto = UpdateReportDto;
//# sourceMappingURL=update-report.dto.js.map