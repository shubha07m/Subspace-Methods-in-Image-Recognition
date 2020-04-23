clear all;
clc;
load facedata % Loading the original data set (1)

%% Converting new images to a new matrix similar to original data set
%% Appending the new data with the original data set (2)
newfaces = [];
for i = 1:10
x = imread(['el',num2str(i),'.jpg']);
x = rgb2gray(x)
x = imresize(x,[1,400])
newfaces =[newfaces;x];
end
newfaces = rescale(newfaces,0,1);
newids = 999*ones(10,1);
faces = [faces; newfaces];
ids = [ids; newids];

%% Computing PCA (3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[A1, s, lat]=pca(faces);

% dimension of each figure
h=20; w=20;

%% Plotting PCA component as per significance and energy (4)
figure(1); 
subplot(1,2,1); grid on; hold on; stem(lat, '.');title('Principle component significance') 
f_eng=lat.*lat; 
subplot(1,2,2); grid on; hold on; plot(cumsum(f_eng)/sum(f_eng), '.-'); title('PCA energy distribution')

%% Plotting Eigen Faces (5)

figure(2)
for k=1:10
    subplot(2,5,k); colormap('gray'); imagesc(reshape(A1(:,k), [h, w]));
    title(sprintf('EigenFace:%d', k));
end

%% top 16 feature selected & shown feature distances between 200 images (6)
kd=16; nface=200;

x=faces*A1(:, 1:kd); 
f_dist = pdist2(x(1:nface,:), x(1:nface,:));
figure(3); 
imagesc(f_dist); colormap('gray'), title('Feature distance between 200 images');

%% calculating true positive, false positive, true negative, false negative (7)
%% upto 7 matching pair , rest are non matching

d0 = f_dist(1:7,1:7); d1=f_dist(8:end, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40);

%% plotting the ROC curve for EigenFace detection (8)
figure(4); hold  on; grid on;
plot(fp./(tn+fp), tp./(tp+fn), '.-r', 'DisplayName', 'tpr-fpr color, data set 1');
xlabel('fpr'); ylabel('tpr'); title('EigenFace recognition performance for 16 features');
legend('kd=16');

%% Plotting of image point in 3D feature space Eigenface separation in feature space (9 & 10)
figure(5); hold on; grid on; 
styl = ['*r'; 'ob'; '+k'; '^m'];
for k=1:4
    figure(5); offs = find(ids==k); plot3(x(offs, 1), x(offs,2), x(offs, 3), styl(k,:) );title('Eigenface separation in feature space');
    figure(6); subplot(2,2,k); imagesc(reshape(faces(offs(1),:), [h, w])); colormap('gray');title(sprintf('EigenFace person:%d', k));
end
%% End of Eigenface implementation (11)%%
%%%%%%%%%%%%%%%%%%%%%%%%%
%%

%% Fisherface implementation (1)

n_face = 1200; n_subj = length(unique(ids(1:n_face))); 
%eigenface kd
kd = 32;
opt.Fisherface = 1; 

%% LDA computation (2)

[A2, lat]=getLDA(faces(1:n_face,:)*A1(:,1:kd),ids(1:n_face));

%% feature distance calculation eigenface and FisherFace (3)
x1 = faces*A1(:,1:kd); 
f_dist1 = pdist2(x1(1:7,:), x1);
% fisherface feature distance calculation
x2 = faces*A1(:,1:kd)*A2; 
f_dist2 = pdist2(x2(1:7,:), x2);

%%
%% scaling the image to kd feature size (4)
eigface = eye(400)*A1(:,1:kd);
fishface = eye(400)*A1(:,1:kd)*A2;

%% Showing total six EigenFace and FisherFace (5)
for k=1:6
   figure(7);
   subplot(2,4,k); imagesc(reshape(eigface(:,k),[20, 20])); colormap('gray');
   title(sprintf('EigenFace:%d', k)); 
   figure(8);
   subplot(2,4,k); imagesc(reshape(fishface(:,k),[20, 20])); colormap('gray');
   title(sprintf('FisherFace:%d', k)); 
end

%% End of Fisherface implementation %% (6)
%%%%%%%%%%%%%%%%%%%%%%%%%


%% Comparison of EigenFace and FisherFace performance (7)
figure(9); grid on; hold on;
% for subj=1
d0 = f_dist1(1:7,1:7); d1=f_dist1(1:7, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40); 
plot(fp./(tn+fp), tp./(tp+fn), '.-k', 'DisplayName', 'eigenface kd=32');

d0 = f_dist2(1:7,1:7); d1=f_dist2(1:7, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40); 
plot(fp./(tn+fp), tp./(tp+fn), '.-r', 'DisplayName', 'fisher kd=32');

xlabel('fpr'); ylabel('tpr'); title(sprintf('eigen vs fisher face recog: %d people, %d faces',n_subj, n_face));
legend('eigen kd=32', 'fisher kd=32'); axis([0 0.25 0 1]);

return;