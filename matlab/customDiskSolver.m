function [forces_final,alphas_final,img_final]=customDiskSolver(forces0,beta,fsigma,rm,z,filename)
verbose=true;
fitoptions = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt','MaxIter',100,'MaxFunEvals',400,'TolFun',0.01,'Display','final-detailed');
fitoptions.FunctionTolerance=1.0e-7;

forceImage=im2double(rgb2gray(imread(filename)));
norm=max(forceImage(:));
forceImage=forceImage/norm;

maskradius=0.98;
scaling=1.0;

%pres=particle;
%disp(['processing file ',fileName, 'containing ' ,num2str(N), 'particles']); %status indicator
%display(['fitting force(s) to particle ',num2str(n)]); %status indicator
if (z > 0 )
%                     %This is the Camera Image
%   %template = im2double(particle(n).forceImage);
             template = imresize(forceImage,scaling);

%                     template = template-0.1;
%                     %template = (template -0.2); %fine tunes the image, this should happen in preprocessing!
%                     template = template.*(template > 0); %fine tunes the image, this should happen in preprocessing!
%                     template = template*3; %fine tunes the image, this should happen in preprocessing!

            %size of the force image
            px = size(template,1);

            %plot the experimental image that is going to be fitted
            %onto
            if verbose
                subplot(1,2,1)
                imshow(template);
            end

            %Create initial guesses for each contact force, based
            %on the gradient squared in the contact region (also
            %better guess lower than too high)
            forces = forces0;


            alphas = zeros(z,1);
            [alphas,forces] = forceBalance(forces,alphas,beta);

            cx=px/2;cy=px/2;ix=px;iy=px;r=maskradius*px;
            [x,y]=meshgrid(-(cx-1):(ix-cx),-(cy-1):(iy-cy));
            c_mask=((x.^2+y.^2)<=r^2);

            func = @(par) joForceImg(z, par(1:z),par(z+1:z+z), beta(1:z), fsigma, rm, px, verbose,par(z+z+1));
            %+par(2*z+1); %this is the function I want to fit (i.e. synthetic stres image), the fitting paramters are in vector par

            err = @(par) real(sum(sum( ( c_mask.*(template-func(par)).^2) )));
            %BUG: for some reason I sometimes get imaginary results, this should not happen


            p0 = zeros(2*z+1, 1);
            p0(1:z) = forces;
            p0(z+1:2*z) = alphas;
            p0(z+z+1) = 1.0;

            %Do the fit, will also work with other solvers
            %TODO: make a user defined option to select between
            %different solvers
            p=lsqnonlin(err,p0,[],[],fitoptions);

            %get back the result from fitting
            forces = p(1:z);
            alphas = p(z+1:z+z);
            artificialOffset = p(z+z+1);
            %resudual
            %fitError = err(p);


            %since the image gnerator also forces force balance we
            %have to explicitly do it once more to the values we
            %are gonna save
            [alphas,forces] = forceBalance(forces,alphas,beta);

             %generate an image with the fitted parameters
            imgFit = joForceImg(z, forces, alphas, beta, fsigma, rm, px*(1/scaling), verbose, artificialOffset);

           % disp(forces)
           % disp(alphas)
           % disp(fitError)

end
forces_final=forces;
alphas_final=alphas;
img_final=imgFit;
end