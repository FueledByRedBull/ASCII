#include "core/cell_stats.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace ascii;

int main() {
    int failures = 0, checked_cells = 0;
    float maximum_mean_error = 0, maximum_variance_error = 0, maximum_edge_error = 0;
    for (int cell_width : {1,4,8,17}) {
        for (int cell_height : {1,5,16,31}) {
            for (int pattern = 0; pattern < 8; ++pattern) {
                const bool full_frame_reference = cell_width==8 && cell_height==16;
                FloatImage image(full_frame_reference ? 960 : cell_width*5+1,
                                 full_frame_reference ? 640 : cell_height*4+2);
                FloatImage squared(image.width(),image.height());
                EdgeData edges;
                edges.magnitude=FloatImage(image.width(),image.height());
                edges.edge_mask.resize(image.size_in_elements());
                for (int y=0;y<image.height();++y) {
                    for (int x=0;x<image.width();++x) {
                        float value=0;
                        if(pattern==1) value=1;
                        if(pattern==2) value=.5f;
                        if(pattern==3) value=.5f+((x+y)%2)*1e-5f;
                        if(pattern==4) value=static_cast<float>(x)/image.width();
                        if(pattern==5) value=(x+y)%2 ? 1.0f : 0.0f;
                        if(pattern==6) value=((x*127+y*73+x*y*29)%251)/250.0f;
                        if(pattern==7) value=(x+y)%97==0 ? 1e-8f : 1.0f;
                        image.set(x,y,value);
                        squared.set(x,y,value*value);
                        edges.magnitude.set(x,y,1.25f*value);
                        edges.edge_mask[static_cast<size_t>(y)*image.width()+x]=(x+y)%3==0;
                    }
                }
                // Compare against the former prefix-sum algorithm. Its large
                // corner subtraction can differ from local accumulation by a
                // few float ulps, so variance has an explicit absolute bound.
                const IntegralImage mean_reference(image), square_reference(squared), edge_reference(edges.magnitude);
                CellStatsAggregator::Config config;
                config.cell_width=cell_width;
                config.cell_height=cell_height;
                config.enable_orientation_histogram=false;
                config.enable_frequency_signature=false;
                config.enable_texture_signature=false;
                const CellStatsAggregator aggregator(config);
                const auto cells=aggregator.compute(image,edges);
                const int cols=aggregator.grid_cols(image.width()), rows=aggregator.grid_rows(image.height());
                for(int row=0;row<rows;++row) {
                    for(int col=0;col<cols;++col) {
                        const int x0=col*cell_width,y0=row*cell_height;
                        const int x1=std::min(x0+cell_width,image.width()),y1=std::min(y0+cell_height,image.height());
                        const float expected_mean=mean_reference.mean(x0,y0,x1,y1);
                        const float expected_variance=std::max(0.0f,square_reference.mean(x0,y0,x1,y1)-expected_mean*expected_mean);
                        const float expected_edge=edge_reference.mean(x0,y0,x1,y1);
                        float peak=0;
                        int occupied=0;
                        for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) {
                            peak=std::max(peak,edges.magnitude.get(x,y));
                            occupied+=edges.is_edge(x,y);
                        }
                        const auto& cell=cells[static_cast<size_t>(row)*cols+col];
                        const float mean_error=std::abs(cell.mean_luminance-expected_mean);
                        const float variance_error=std::abs(cell.luminance_variance-expected_variance);
                        const float edge_error=std::abs(cell.edge_strength-expected_edge);
                        maximum_mean_error=std::max(maximum_mean_error,mean_error);
                        maximum_variance_error=std::max(maximum_variance_error,variance_error);
                        maximum_edge_error=std::max(maximum_edge_error,edge_error);
                        const float occupancy=static_cast<float>(occupied)/((x1-x0)*(y1-y0));
                        bool valid=std::isfinite(cell.mean_luminance) && std::isfinite(cell.luminance_variance) &&
                            std::isfinite(cell.edge_strength) && cell.luminance_variance>=0 && mean_error<=1e-6f &&
                            variance_error<=1e-6f && edge_error<=1e-6f && cell.edge_strength_max==peak &&
                            cell.edge_occupancy==occupancy &&
                            cell.is_edge_cell==(occupancy>=config.edge_threshold || peak>=config.edge_threshold) &&
                            std::abs(cell.local_contrast*cell.local_contrast-cell.luminance_variance)<=1e-6f;
                        if(pattern<=2) valid &= cell.luminance_variance==0;
                        if(!valid && ++failures<8) std::cerr<<"Aggregation mismatch: "<<cell_width<<'x'<<cell_height<<" pattern="<<pattern<<'\n';
                        ++checked_cells;
                    }
                }
            }
        }
    }
    if(!CellStatsAggregator{}.compute(FloatImage{},{}).empty()) ++failures;
    std::cout<<"Checked "<<checked_cells<<" cells; max mean="<<maximum_mean_error<<" variance="<<maximum_variance_error
             <<" edge="<<maximum_edge_error<<" failures="<<failures<<'\n';
    return failures!=0;
}
