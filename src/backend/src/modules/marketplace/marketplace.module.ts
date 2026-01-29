import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { MarketplaceService } from './marketplace.service';
import { MarketplaceResolver } from './marketplace.resolver';
import { CalibrationListing } from './entities/calibration-listing.entity';
import { License } from './entities/license.entity';
import { Review } from './entities/review.entity';
import { AuthModule } from '../auth/auth.module';
import { MarketplaceController } from './marketplace.controller';

@Module({
    imports: [
        TypeOrmModule.forFeature([CalibrationListing, License, Review]),
        AuthModule,
    ],
    controllers: [MarketplaceController],
    providers: [MarketplaceService, MarketplaceResolver],
    exports: [MarketplaceService],
})
export class MarketplaceModule { }
